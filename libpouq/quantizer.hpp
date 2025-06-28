#pragma once

#include <omp.h>

#include "clusterer.hpp"
#include "optimizer.hpp"

namespace pouq {

inline void set(uint8_t *data, size_t index, size_t n) {
  n &= 3;
  const size_t i      = index * 2 / 8;
  const size_t offset = index * 2 % 8;
  data[i] &= ~(3 << offset);
  data[i] |= n << offset;
}

inline std::tuple<size_t, size_t, size_t, size_t> get(uint8_t byte) {
  return {
      byte & 3,
      byte >> 2 & 3,
      byte >> 4 & 3,
      byte >> 6 & 3,
  };
}

class Quantizer {

public:
  explicit Quantizer(size_t groups) : dim_(groups) {}

  void train(const float *data, size_t size) {
    std::vector step_size_(dim_, std::vector<float>(4));
    std::vector lower_bound_(dim_, std::vector<float>(4));
    cid_.resize(size / dim_, std::vector<uint8_t>(dim_ / 4));
    code_.resize(size / dim_, std::vector<uint8_t>(dim_ / 4));

#pragma omp parallel for
    for (size_t d = 0; d < dim_; d++) {
      const auto data_freq_map = count_freq(data, size, d);
      const auto bounds        = cluster(4, data_freq_map);

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

          const auto [opt_lower, opt_upper] = optimise(3, lower, upper, data_start, data_end);
          lower                             = opt_lower;
          upper                             = opt_upper;
        }
        lower_bound_[d][i] = lower;
        if (lower == upper) {
          step_size_[d][i] = 1.0;
        } else {
          step_size_[d][i] = (upper - lower) / 3.0f;
        }
      }

      for (size_t i = d; i < size; i += dim_) {
        const auto it = std::upper_bound(
            bounds.begin(), bounds.end(), data[i], [](const float rhs, const std::pair<float, float> &lhs) -> bool {
              return rhs < lhs.first;
            });
        const size_t c = it - bounds.begin() - 1;
        const float  x = std::clamp((data[i] - lower_bound_[d][c]) / step_size_[d][c] + 0.5f, 0.0f, 3.0f);
        set(cid_[i / dim_].data(), i % dim_, c);
        set(code_[i / dim_].data(), i % dim_, x);
      }
    }

    lower_bound_128.resize(dim_ / 4, std::vector(256, std::vector<float>(4)));
    step_size_128.resize(dim_ / 4, std::vector(256, std::vector<float>(4)));

#pragma omp parallel for
    for (size_t i = 0; i < dim_; i += 4) {
      for (size_t j = 0; j < 256; j++) {
        auto [x0, x1, x2, x3] = get(j);

        lower_bound_128[i / 4][j][0] = lower_bound_[i + 0][x0];
        lower_bound_128[i / 4][j][1] = lower_bound_[i + 1][x1];
        lower_bound_128[i / 4][j][2] = lower_bound_[i + 2][x2];
        lower_bound_128[i / 4][j][3] = lower_bound_[i + 3][x3];

        step_size_128[i / 4][j][0] = step_size_[i + 0][x0];
        step_size_128[i / 4][j][1] = step_size_[i + 1][x1];
        step_size_128[i / 4][j][2] = step_size_[i + 2][x2];
        step_size_128[i / 4][j][3] = step_size_[i + 3][x3];
      }
    }
  }

  float l2distance(const float *data, size_t offset) const {
    float result = 0;
    for (size_t i = 0; i < dim_; i += 4) {
      auto cid_byte         = cid_[offset / dim_][i / 4];
      auto [x0, x1, x2, x3] = get(code_[offset / dim_][i / 4]);

      auto d0 = static_cast<float>(x0) * step_size_128[i / 4][cid_byte][0] + lower_bound_128[i / 4][cid_byte][0] -
                data[i + 0];
      auto d1 = static_cast<float>(x1) * step_size_128[i / 4][cid_byte][1] + lower_bound_128[i / 4][cid_byte][1] -
                data[i + 1];
      auto d2 = static_cast<float>(x2) * step_size_128[i / 4][cid_byte][2] + lower_bound_128[i / 4][cid_byte][2] -
                data[i + 2];
      auto d3 = static_cast<float>(x3) * step_size_128[i / 4][cid_byte][3] + lower_bound_128[i / 4][cid_byte][3] -
                data[i + 3];

      result += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    return result;
  }

private:
  size_t                                       dim_ = 0;
  std::vector<std::vector<std::vector<float>>> lower_bound_128;
  std::vector<std::vector<std::vector<float>>> step_size_128;
  std::vector<std::vector<uint8_t>>            cid_;
  std::vector<std::vector<uint8_t>>            code_;

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