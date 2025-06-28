#pragma once

#include <omp.h>

#include "../clusterer.hpp"
#include "../optimizer.hpp"

class POUQ4 {
public:
  explicit POUQ4(size_t sub) : dim_(sub) {}

  void train(const float *data, size_t size) {
    size_          = size;
    step_size_     = new float[dim_ * (1 << 2)];
    lower_bound_   = new float[dim_ * (1 << 2)];
    cid_           = new uint8_t[(2 * size_ + 7) / 8];
    code_          = new uint8_t[(2 * size_ + 7) / 8];
    const auto div = static_cast<float>((1 << 2) - 1);

#pragma omp parallel for default(none) shared(data, div)
    for (size_t group = 0; group < dim_; group++) {
      const auto data_freq_map = count_freq(data, group);
      const auto bounds        = clusterer(1 << 2, data_freq_map);
      const auto offset        = group * (1 << 2);

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

          const auto [opt_lower, opt_upper] = optimizer(div, data_start, data_end);
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
      for (size_t i = group; i < size_; i += dim_) {
        const float d  = data[i];
        const auto  it = std::upper_bound(
            bounds.begin(), bounds.end(), d, [](const float rhs, const std::pair<float, float> &lhs) -> bool {
              return rhs < lhs.first;
            });
        const size_t c = it - bounds.begin() - 1;
        set(cid_, i, c);
        const float x = std::clamp((d - lower_bound_[offset + c]) / step_size_[offset + c] + 0.5f, 0.0f, div);
        set(code_, i, static_cast<size_t>(x));
      }
    }
  }

  float l2distance(const float *data, size_t data_index) const {
    float result = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
      float dif = data[i] - operator[](i + data_index);
      result += dif * dif;
    }
    return result;
  }

  float operator[](size_t i) const {
    const size_t group  = i % dim_;
    const size_t offset = get(cid_, i) + group * (1 << 2);
    const size_t x      = get(code_, i);
    return lower_bound_[offset] + step_size_[offset] * static_cast<float>(x);
  }

  size_t size() const { return size_; }

  ~POUQ4() {
    delete[] lower_bound_;
    delete[] step_size_;
    delete[] cid_;
    delete[] code_;
  }

private:
  size_t   size_        = 0;
  size_t   dim_         = 0;
  float   *lower_bound_ = nullptr;
  float   *step_size_   = nullptr;
  uint8_t *cid_         = nullptr;
  uint8_t *code_        = nullptr;

  static inline pouq::KrangeClusterer clusterer;
  static inline pouq::PSOptimizer     optimizer;

  std::vector<std::pair<float, size_t>> count_freq(const float *data, const size_t group) const {
    std::vector<float> sorted_data;
    sorted_data.reserve(size_ / dim_);
    for (size_t i = group; i < size_; i += dim_) {
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

  inline void set(uint8_t *data, size_t index, size_t n) {
    n &= (1 << 2) - 1;
    const size_t pos = index * 2;
    for (size_t bit = 0; bit < 2; ++bit) {
      const size_t i      = (pos + bit) / 8;
      const size_t offset = (pos + bit) % 8;
      if (n & 1 << bit) {
        data[i] |= 1 << offset;
      } else {
        data[i] &= ~(1 << offset);
      }
    }
  }

  inline size_t get(const uint8_t *data, size_t index) const {
    const size_t pos    = index * 2;
    size_t       result = 0;
    for (size_t bit = 0; bit < 2; ++bit) {
      const size_t i      = (pos + bit) / 8;
      const size_t offset = (pos + bit) % 8;
      if (data[i] & 1 << offset) {
        result |= 1 << bit;
      }
    }

    return result;
  }
};
