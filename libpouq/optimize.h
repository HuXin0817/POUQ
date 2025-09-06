#pragma once

#include <cassert>
#include <random>

#include "simd/loss.h"

struct Parameter {
  int max_iter = 100;
  int particle_count = 50;
  float scale_factor = 0.1f;
  float init_inertia = 0.9f;
  float final_inertia = 0.4f;
  float c1 = 1.5f;
  float c2 = 1.5f;
};

std::pair<float, float>
optimize(float div,
         float init_lower,
         float init_upper,
         const float* data_map,
         const int* freq_map,
         int size,
         const Parameter& parameter,
         bool do_count_freq);
