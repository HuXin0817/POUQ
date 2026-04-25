#include "optimizer.h"

#include <algorithm>
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

  xsimd::batch<float> zero_batch = 0.0f;
  xsimd::batch<float> level_batch = level;
  xsimd::batch<float> lower_batch = lower;
  xsimd::batch<float> step_size_batch = step_size;

  for (const xsimd::batch<float>& d : batch_data) {
    xsimd::batch<float> code = xsimd::round((d - lower) / step_size);
    code = xsimd::clip(code, zero_batch, level_batch);
    xsimd::batch<float> rebuild = xsimd::fma(code, step_size_batch, lower_batch);
    xsimd::batch<float> diff = rebuild - d;
    result += xsimd::reduce_add(diff * diff);
  }

  for (const float d : extra) {
    float code = std::roundf((d - lower) / step_size);
    code = std::clamp(code, 0.0f, level);
    float rebuild = lower + step_size * code;
    float diff = rebuild - d;
    result += diff * diff;
  }

  return result;
}

struct Context {
  float level;
  std::vector<xsimd::batch<float>> batch_data;
  std::span<float> extra;
};

double LossWrapper(const std::vector<double>& x, std::vector<double>&, void* data) {
  const Context& context = *static_cast<Context*>(data);
  return Loss(context.level, x[0], x[1], context.batch_data, context.extra);
}

std::pair<float, float> Optimizer::Optimize(const std::span<float>& data,
                                            nlopt::algorithm algorithm,
                                            int maxeval,
                                            float scale_factor) {
  float init_lower = data.front();
  float init_upper = data.back();
  float width = (init_upper - init_lower) * std::clamp(scale_factor, 0.0f, 0.5f);

  const std::vector<double> lower_bound = {init_lower - width, init_upper - width};
  const std::vector<double> upper_bound = {init_lower + width, init_upper + width};

  Context context{.level = level_};

  size_t i = 0;
  for (; i < data.size(); i += xsimd::batch<float>::size) {
    context.batch_data.push_back(xsimd::batch<float>::load_unaligned(data.data() + i));
  }

  if (i > data.size()) {
    i -= xsimd::batch<float>::size;
    context.extra = {data.begin() + i, data.end()};
  }

  nlopt::opt opt(algorithm, 2);
  opt.set_lower_bounds(lower_bound);
  opt.set_upper_bounds(upper_bound);
  opt.set_min_objective(LossWrapper, &context);
  opt.set_maxeval(maxeval);

  std::vector<double> x = {init_lower, init_upper};
  double minf;
  opt.optimize(x, minf);

  return {x[0], x[1]};
}

}  // namespace pouq::optimize