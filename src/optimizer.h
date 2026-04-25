#pragma once

#include <span>
#include <utility>

namespace pouq::optimize {

class Optimizer {
 public:
  Optimizer(float level) : level_(level) {}

  ~Optimizer() = default;

  std::pair<float, float> Optimize(const std::span<float>& data);

 private:
  float level_;
  int maxeval_ = 5000;
  float scale_factor_ = 0.1f;
};

}  // namespace pouq::optimize