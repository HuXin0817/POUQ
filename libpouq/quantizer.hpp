#pragma once

#include <omp.h>

#include "bitmap.hpp"
#include "clusterer.hpp"
#include "optimizer.hpp"

namespace pouq {

class Quantizer {
public:
  explicit Quantizer(size_t c_bit, size_t groups) : c_bit_(c_bit), dim_(groups) {}

  void train(const float *data, size_t size) {
    step_size_     = new float[dim_ * (1 << c_bit_)];
    lower_bound_   = new float[dim_ * (1 << c_bit_)];
    cid_           = new uint8_t[(c_bit_ * size + 7) / 8];
    code_          = new uint8_t[(c_bit_ * size + 7) / 8];
    const auto div = static_cast<float>((1 << c_bit_) - 1);

#pragma omp parallel for
    for (size_t group = 0; group < dim_; group++) {
      const auto data_freq_map = count_freq(data, size, group);
      const auto bounds        = cluster(1 << c_bit_, data_freq_map);
      const auto offset        = group * (1 << c_bit_);

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
        lower_bound_[offset + i] = lower;
        if (lower == upper || div == 0.0f) {
          step_size_[offset + i] = 1.0;
        } else {
          step_size_[offset + i] = (upper - lower) / div;
        }
      }

      static_cast<std::vector<std::pair<float, size_t>>>(data_freq_map).clear();
      for (size_t i = group; i < size; i += dim_) {
        const float d  = data[i];
        const auto  it = std::upper_bound(
            bounds.begin(), bounds.end(), d, [](const float rhs, const std::pair<float, float> &lhs) -> bool {
              return rhs < lhs.first;
            });
        const size_t c = it - bounds.begin() - 1;
        bitmap::set(cid_, i, c, c_bit_);
        const float x = std::clamp((d - lower_bound_[offset + c]) / step_size_[offset + c] + 0.5f, 0.0f, div);
        bitmap::set(code_, i, static_cast<size_t>(x), c_bit_);
      }
    }
  }

  float operator[](size_t i) const {
    const size_t group  = i % dim_;
    const size_t offset = bitmap::get(cid_, i, c_bit_) + group * (1 << c_bit_);
    const size_t x      = bitmap::get(code_, i, c_bit_);
    return lower_bound_[offset] + step_size_[offset] * static_cast<float>(x);
  }

  ~Quantizer() {
    delete[] lower_bound_;
    delete[] step_size_;
    delete[] cid_;
    delete[] code_;
  }

private:
  size_t   c_bit_       = 0;
  size_t   dim_         = 0;
  float   *lower_bound_ = nullptr;
  float   *step_size_   = nullptr;
  uint8_t *cid_         = nullptr;
  uint8_t *code_        = nullptr;

  std::vector<std::pair<float, size_t>> count_freq(const float *data, size_t size, const size_t group) const {
    std::vector<float> sorted_data;
    sorted_data.reserve(size / dim_);
    for (size_t i = group; i < size; i += dim_) {
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
};

}  // namespace pouq