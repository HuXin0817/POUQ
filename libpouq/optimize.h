#pragma once

#include <assert.h>
#include <float.h>
#include <math.h>

#include "def.h"
#include "util.h"

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
         bool do_count_freq);
