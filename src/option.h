#pragma once

#include <nlopt.hpp>

namespace pouq {

struct TrainOption {
  bool use_optimizer = true;
  nlopt::algorithm algorithm = nlopt::GN_ISRES;
  int maxeval = 100;
  float scale_factor = 0.1f;
};

}  // namespace pouq