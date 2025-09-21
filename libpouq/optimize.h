#pragma once

#include <assert.h>

typedef struct {
  int max_iter;
  int particle_count;
  float scale_factor;
  float init_inertia;
  float final_inertia;
  float c1;
  float c2;
} Parameter;

typedef struct {
  float lower;
  float upper;
} Bound;

Bound
optimize(float init_lower,
         float init_upper,
         const float* data_map,
         const int* freq_map,
         int size,
         const Parameter parameter,
         int do_count_freq);
