#pragma once

#include <omp.h>

#include "bitmap.hpp"
#include "clusterer.hpp"
#include "optimizer.hpp"

namespace pouq {

std::vector<std::pair<float, size_t>> count_freq(const float *data, size_t size, size_t d, size_t dim) {
  std::vector<float> sorted_data;
  sorted_data.reserve(size / dim);
  for (size_t i = d; i < size; i += dim) {
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

class Quantizer {
public:
  explicit Quantizer(size_t nbit, size_t dim) : nbit_(nbit), dim_(dim) {}

  void train(const float *data, size_t size) {
    codebook       = new std::pair<float, float>[dim_ * (1 << nbit_)];
    codes_         = new uint8_t[(nbit_ * size * 2 + 7) / 8];
    const auto div = static_cast<float>((1 << nbit_) - 1);

#pragma omp parallel for
    for (size_t d = 0; d < dim_; d++) {
      const auto data_freq_map = count_freq(data, size, d, dim_);
      const auto bounds        = cluster(1 << nbit_, data_freq_map);
      const auto offset        = d * (1 << nbit_);

      for (size_t i = 0; i < bounds.size(); i++) {
        auto [lower, upper] = bounds[i];
        if (lower < upper) {
          auto data_start = std::lower_bound(data_freq_map.begin(),
              data_freq_map.end(),
              lower,
              [](const std::pair<float, size_t> &lhs, const float rhs) -> bool { return lhs.first < rhs; });
          auto data_end   = std::upper_bound(data_freq_map.begin(),
              data_freq_map.end(),
              upper,
              [](const float rhs, const std::pair<float, size_t> &lhs) -> bool { return rhs < lhs.first; });

          const auto [opt_lower, opt_upper] = optimise(div, lower, upper, data_start, data_end);
          lower                             = opt_lower;
          upper                             = opt_upper;
        }
        if (lower == upper || div == 0.0f) {
          codebook[offset + i] = {lower, 1.0f};
        } else {
          codebook[offset + i] = {lower, (upper - lower) / div};
        }
      }

      static_cast<std::vector<std::pair<float, size_t>>>(data_freq_map).clear();
      for (size_t i = d; i < size; i += dim_) {
        const float d  = data[i];
        const auto  it = std::upper_bound(
            bounds.begin(), bounds.end(), d, [](const float rhs, const std::pair<float, float> &lhs) -> bool {
              return rhs < lhs.first;
            });
        const size_t c                      = it - bounds.begin() - 1;
        const auto [lower_bound, step_size] = codebook[offset + c];
        const float x                       = std::clamp((d - lower_bound) / step_size + 0.5f, 0.0f, div);

        bitmap::set(codes_, 2 * i, c, nbit_);
        bitmap::set(codes_, 2 * i + 1, x, nbit_);
      }
    }
  }

  float operator[](size_t i) const {
    const size_t d                      = i % dim_;
    const size_t offset                 = bitmap::get(codes_, 2 * i, nbit_);
    const size_t x                      = bitmap::get(codes_, 2 * i + 1, nbit_);
    const auto [lower_bound, step_size] = codebook[offset + d * (1 << nbit_)];
    return lower_bound + step_size * static_cast<float>(x);
  }

  ~Quantizer() {
    delete[] codebook;
    delete[] codes_;
  }

private:
  size_t                   nbit_   = 0;
  size_t                   dim_     = 0;
  std::pair<float, float> *codebook = nullptr;
  uint8_t                 *codes_   = nullptr;
};

}  // namespace pouq