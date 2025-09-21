#pragma once

#include <float.h>

int
segment(int k,
        const float* data_map,
        const int* freq_map,
        int size,
        int do_count_freq,
        float* lowers,
        float* uppers);
