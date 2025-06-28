#pragma once

#include <omp.h>

#include "../bitmap.hpp"
#include "../clusterer.hpp"
#include "../optimizer.hpp"

namespace posq {

class POSQ8 {
public:
  explicit POSQ8(size_t dim) : dim_(dim) {}

  void train(const float *data, size_t size) {
    // size_        = size;
    codebook_ = new float[dim_ * (1 << 4) * 2];
    codes_    = new uint8_t[size];

#pragma omp parallel for
    for (size_t group = 0; group < dim_; group++) {
      const auto data_freq_map = count_freq(data, size, group);
      const auto bounds        = clusterer(1 << 4, data_freq_map);
      const auto offset        = group * (1 << 4);

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
        auto off_       = (offset + i) << 1;
        codebook_[off_] = lower;
        if (lower == upper) {
          codebook_[off_ + 1] = 1.0;
        } else {
          codebook_[off_ + 1] = (upper - lower) / div;
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
        set(codes_, 2 * i, c);
        auto        off_ = (offset + c) << 1;
        const float x    = std::clamp((d - codebook_[off_]) / codebook_[off_ + 1] + 0.5f, 0.0f, div);
        set(codes_, 2 * i + 1, x);
      }
    }
  }

  float l2distance(const float *data, size_t n) const {
    float dis = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
      uint8_t v      = codes_[n + i];
      size_t  off    = 2 * ((v & 0xF) + i * (1 << 4));
      float   decode = codebook_[off] + codebook_[off + 1] * (v >> 4 & 0xF);
      float   diff   = data[i] - decode;
      dis += diff * diff;
    }
    return dis;
  }

  // size_t size() const { return size_; }

  ~POSQ8() {
    delete[] codebook_;
    delete[] codes_;
  }

private:
  // size_t   size_        = 0;
  size_t dim_ = 0;
  // float *codebook_ = nullptr;
  // float *codebook_   = nullptr;
  float *codebook_ = nullptr;
  // uint8_t *cid_         = nullptr;
  // uint8_t *code_        = nullptr;
  uint8_t *codes_ = nullptr;

  static inline KrangeClusterer clusterer;
  static inline PSOptimizer     optimizer;

  static constexpr auto div = static_cast<float>((1 << 4) - 1);

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

  // float at(size_t i) const {
  //   const auto   v      = codes_[i];
  //   const size_t group  = i % dim_;
  //   const size_t offset = 2 * ((v & 0xF) + group * (1 << 4));
  //   return codebook_[offset] + codebook_[offset + 1] * static_cast<float>(v >> 4 & 0xF);
  // }

  void set(uint8_t *data, size_t index, size_t n) {
    n &= (1 << 4) - 1;
    const size_t pos = index * 4;
    for (size_t bit = 0; bit < 4; ++bit) {
      const size_t i      = (pos + bit) / 8;
      const size_t offset = (pos + bit) % 8;
      if (n & 1 << bit) {
        data[i] |= 1 << offset;
      } else {
        data[i] &= ~(1 << offset);
      }
    }
  }

  // std::pair<size_t, size_t> get_pair(const uint8_t *data, size_t index) const {
  //   const size_t pos = index * 4;
  //   const size_t byte_idx = pos / 8;
  //   const size_t bit_offset = pos % 8;
  //   const uint8_t byte_val = data[byte_idx];
  //
  //   const size_t b1 = (byte_val >> bit_offset) & 0xF;
  //   const size_t b2 = (byte_val >> (bit_offset + 4)) & 0xF;
  //   return {b1, b2};
  // }
};

}  // namespace posq