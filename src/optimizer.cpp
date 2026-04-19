#include "optimizer.h"

#include <cmath>
#include <nlopt.hpp>
#include <xsimd/xsimd.hpp>

namespace pouq::optimize {

float Loss(float level,
           float lower,
           float upper,
           const std::vector<xsimd::batch<float>>& batch_data,
           const std::span<float>& extra) {
  float result = 0.0f;

  float step_size = (upper - lower) / level;

  for (const xsimd::batch<float>& d : batch_data) {
    xsimd::batch<float> code = xsimd::round((d - lower) / step_size);
    code = xsimd::clip(code, xsimd::batch{0.0f}, xsimd::batch{level});
    xsimd::batch<float> qdata = xsimd::fma(code, xsimd::batch{step_size}, xsimd::batch{lower});
    xsimd::batch<float> diff = qdata - d;
    result += xsimd::reduce_add(diff * diff);
  }

  for (const float d : extra) {
    float code = std::roundf((d - lower) / step_size);
    code = std::clamp(code, 0.0f, level);
    float qdata = lower + step_size * code;
    float diff = qdata - d;
    result += diff * diff;
  }

  return result;
}

using Package = std::tuple<float, std::vector<xsimd::batch<float>>, std::span<float>>;

double LossWrapper(const std::vector<double>& x, std::vector<double>&, void* data) {
  const Package& unpack = *static_cast<Package*>(data);
  return Loss(std::get<0>(unpack), x[0], x[1], std::get<1>(unpack), std::get<2>(unpack));
}

std::pair<float, float> Optimizer::Optimize(const std::span<float>& data) {
  float init_lower = data.front();
  float init_upper = data.back();
  float width = (init_upper - init_lower) * scale_factor;

  const std::vector<double> lower_bound = {init_lower - width, init_upper - width};
  const std::vector<double> upper_bound = {init_lower + width, init_upper + width};

  Package package = {level_, {}, {}};

  size_t i = 0;
  for (; i < data.size(); i += xsimd::batch<float>::size) {
    std::get<1>(package).push_back(xsimd::batch<float>::load_unaligned(data.data() + i));
  }

  if (i > data.size()) {
    i -= xsimd::batch<float>::size;
    std::get<2>(package) = {data.begin() + i, data.end()};
  }

  nlopt::opt opt(nlopt::GN_ISRES, 2);
  opt.set_lower_bounds(lower_bound);
  opt.set_upper_bounds(upper_bound);
  opt.set_min_objective(LossWrapper, &package);
  opt.set_maxeval(max_iter);

  std::vector<double> x = {init_lower, init_upper};
  double minf;
  opt.optimize(x, minf);

  return {x[0], x[1]};
}

}  // namespace pouq::optimize