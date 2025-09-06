#pragma once

#include <cassert>
#include <cfloat>
#include <cmath>

#include "def.h"

float
loss(float div,
     float lower,
     float step,
     const float* data_map,
     const int* freq_map,
     int size,
     bool do_count_freq);
