#pragma once

#include <span>
#include <utility>
#include <xsimd/xsimd.hpp>

namespace pouq::optimize {

class Optimizer {
 public:
  Optimizer(float level) : level_(level) {}

  ~Optimizer() = default;

  std::pair<float, float> Optimize(const std::span<float>& data, int maxeval, float scale_factor);

 private:
  float level_;
};

}  // namespace pouq::optimize