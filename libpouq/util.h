#pragma once

#include <stdlib.h>

float
rand_float(float a, float b);

int
get_sorted_data(const float* data, int size, int d, int dim, float* sorted_data);

int
count_unique(const float* arr, int len);

int
count_freq(const float* sorted_data, int sorted_data_size, float* data_map, int* freq_map);
