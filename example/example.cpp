#include "../libpouq/clusterer.hpp"
#include "../libpouq/utils.hpp"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>

constexpr size_t N = 1e6;

std::vector<std::pair<float, size_t>> count_freq(const float *data, size_t size_, const size_t group, size_t groups_) {
  std::vector<float> sorted_data;
  sorted_data.reserve(size_ / groups_);
  for (size_t i = group; i < size_; i += groups_) {
    sorted_data.push_back(data[i]);
  }
  std::sort(sorted_data.begin(), sorted_data.end());

  float                                 curr_value = sorted_data[0];
  size_t                                count      = 1;
  std::vector<std::pair<float, size_t>> data_freq_map;
  data_freq_map.reserve(sorted_data.size());
  for (size_t i = 1; i < sorted_data.size(); i++) {
    if (sorted_data[i] == curr_value) {
      count++;
    } else {
      data_freq_map.emplace_back(curr_value, count);
      curr_value = sorted_data[i];
      count      = 1;
    }
  }

  data_freq_map.emplace_back(curr_value, count);
  return data_freq_map;
}

std::pair<float, float> mse_bound(float          k,
    const std::vector<std::pair<float, float>>  &bound,
    const std::vector<std::pair<float, size_t>> &data_freq_map) {
  float  mse      = 0.0f;
  float  real_mse = 0.0f;
  size_t p        = 0;

  for (auto [lb, up] : bound) {
    float step_size = (up - lb) / (k - 1.f);
    if (up == lb) {
      step_size = 1.0f;
    }

    size_t start = p;
    size_t cnt   = 0;
    while (data_freq_map[p].first < up) {
      cnt += data_freq_map[p].second;
      p++;
    }
    mse += static_cast<float>(cnt) * step_size * step_size/4.f;
    for (size_t i = start; i < p; i++) {
      auto code   = std::round((data_freq_map[i].first - lb) / step_size);
      auto decode = code * step_size + lb;
      auto dif    = data_freq_map[i].first - decode;
      real_mse += dif * dif * data_freq_map[i].second;
    }
  }
  return {mse, real_mse};
}

template <typename Clusterer>
void run(size_t k, std::vector<float> &data, const std::vector<std::vector<std::pair<float, size_t>>> &data_freq_maps) {
  float mse      = 0.0f;
  float read_mse = 0.0f;

#pragma omp parallel for reduction(+ : mse, read_mse)
  for (size_t i = 0; i < data_freq_maps.size(); i++) {
    const auto &d     = data_freq_maps[i];
    auto        bound = Clusterer()(k, d);
    auto        p     = mse_bound(k, bound, d);
    mse += p.first;
    read_mse += p.second;
  }
  std::cout << mse / data.size() << " " << read_mse / data.size() << std::endl;
}

int main(int argc, char *argv[]) {
  const std::string dataset_name = argv[1];

  auto  d1   = read_fvecs("../data/" + dataset_name + "/" + dataset_name + "_base.fvecs");
  auto &data = d1.first;
  auto  dim  = d1.second;

  std::vector<std::vector<std::pair<float, size_t>>> data_freq_maps(dim);

#pragma omp parallel for
  for (size_t i = 0; i < dim; i++) {
    data_freq_maps[i] = count_freq(data.data(), data.size(), i, dim);
  }

  run<pouq::Clusterer>(16, data, data_freq_maps);
  run<pouq::KmeansClusterer>(16, data, data_freq_maps);
  run<pouq::KrangeClusterer>(16, data, data_freq_maps);

  run<pouq::Clusterer>(256, data, data_freq_maps);
  run<pouq::KmeansClusterer>(256, data, data_freq_maps);
  run<pouq::KrangeClusterer>(256, data, data_freq_maps);
}
