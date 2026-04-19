#pragma once

#include <span>
#include <utility>
#include <xsimd/xsimd.hpp>

namespace pouq::optimize {

class Optimizer {
 public:
  Optimizer(float level) : level_(level) {}

  ~Optimizer() = default;

  std::pair<float, float> Optimize(const std::span<float>& data);

 private:
  float level_;
  uint32_t max_iter = 10000;
  float scale_factor = 0.3f;
};

}  // namespace pouq::optimize