#pragma once

#include <assert.h>
#include <float.h>
#include <math.h>

#include "def.h"

float
loss(float div,
     float lower,
     float step,
     const float* data_map,
     const int* freq_map,
     int size,
     int do_count_freq);
