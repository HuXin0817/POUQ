#pragma once

#include <assert.h>
#include <float.h>
#include <string.h>

#include "def.h"

int
segment(const float* data_map,
        const int* freq_map,
        int size,
        bool do_count_freq,
        float* lowers,
        float* uppers);
