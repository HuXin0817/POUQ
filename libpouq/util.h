#pragma once

#include <vector>

std::vector<float>
get_sorted_data(const float* data, int size, int d, int dim);

std::pair<std::vector<float>, std::vector<int>>
count_freq(const std::vector<float>& sorted_data);
