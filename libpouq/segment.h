#pragma once

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <utility>
#include <vector>

std::vector<std::pair<float, float>>
segment(int k, const float* data_map, const int* freq_map, int size, bool do_count_freq);
