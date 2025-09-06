#pragma once

namespace pouq {

static std::vector<float>
get_sorted_data(const float* data, int size, int d, int dim) {
  std::vector<float> sorted_data;
  sorted_data.reserve(size / dim);
  for (int i = d; i < size; i += dim) {
    sorted_data.push_back(data[i]);
  }
  std::sort(sorted_data.begin(), sorted_data.end());
  return sorted_data;
}

static std::pair<std::vector<float>, std::vector<int>>
count_freq(const std::vector<float>& sorted_data) {
  float curr_value = sorted_data[0];
  int count = 1;
  std::vector<float> data_map;
  std::vector<int> freq_map;
  data_map.reserve(sorted_data.size());
  freq_map.reserve(sorted_data.size());
  for (int i = 1; i < sorted_data.size(); i++) {
    if (sorted_data[i] == curr_value) {
      count++;
    } else {
      data_map.emplace_back(curr_value);
      freq_map.emplace_back(count);
      curr_value = sorted_data[i];
      count = 1;
    }
  }

  data_map.emplace_back(curr_value);
  freq_map.emplace_back(count);
  return {data_map, freq_map};
}

}  // namespace pouq