#pragma once

namespace pouq {

struct TrainOption {
  bool use_optimizer = true;
  int maxeval = 100;
  float scale_factor = 0.1f;
};

}  // namespace pouq