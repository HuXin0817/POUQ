#pragma once

#include "clusterer.hpp"
#include "optimizer.hpp"
#include <immintrin.h>
#include <omp.h>

class POUQ4 {
public:
  explicit POUQ4(size_t sub) : dim_(sub) {}

  void train(const float *data, size_t size) {
    size_              = size;
    step_size_         = new float[dim_ * (1 << 2)];
    lower_bound_       = new float[dim_ * (1 << 2)];
    codes_             = new uint8_t[(4 * size_ + 7) / 8];
    constexpr auto div = static_cast<float>((1 << 2) - 1);

#pragma omp parallel for default(none) shared(data, div)
    for (size_t group = 0; group < dim_; group++) {
      const auto data_freq_map = count_freq(data, group);
      const auto bounds        = pouq::cluster(1 << 2, data_freq_map);
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

          const auto [opt_lower, opt_upper] = pouq::optimise(div, lower, upper, data_start, data_end);
          lower                             = opt_lower;
          upper                             = opt_upper;
        }
        lower_bound_[offset + i] = lower;
        if (lower == upper) {
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
        set(codes_, i * 2, c);
        const float x = std::clamp((d - lower_bound_[offset + c]) / step_size_[offset + c] + 0.5f, 0.0f, div);
        set(codes_, i * 2 + 1, static_cast<size_t>(x));
      }
    }
  }

  float l2distance(const float *data, size_t data_index) const {
    __m256 sum_vec = _mm256_setzero_ps();

    // Main loop: process 8 dimensions at a time
    size_t i = 0;
    for (; i <= dim_ - 8; i += 8) {
      size_t index_start = i + data_index;
      size_t byte_pos    = (index_start * 4) / 8;

      // Load 4 bytes containing codes for 8 dimensions
      uint32_t bytes = *(uint32_t *)&codes_[byte_pos];

      // Extract 8 combined values (4 bits each) into a 256-bit vector
      __m256i bytes_vec = _mm256_set1_epi32(bytes);
      __m256i shifts    = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
      __m256i shifted   = _mm256_srlv_epi32(bytes_vec, shifts);
      __m256i combined  = _mm256_and_si256(shifted, _mm256_set1_epi32(0xF));

      // Extract offset (lower 2 bits) and x (upper 2 bits)
      __m256i offset = _mm256_and_si256(combined, _mm256_set1_epi32(0x3));
      __m256i x      = _mm256_and_si256(_mm256_srli_epi32(combined, 2), _mm256_set1_epi32(0x3));

      // Compute final_offset = offset + i * 4 for each lane
      __m256i i4  = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
      __m256i idx = _mm256_add_epi32(offset, i4);

      // Gather lower_bound_ and step_size_ using indices
      __m256 lb = _mm256_i32gather_ps(lower_bound_, idx, 4);
      __m256 ss = _mm256_i32gather_ps(step_size_, idx, 4);

      // Compute quantized values: lb + ss * x
      __m256 x_float         = _mm256_cvtepi32_ps(x);
      __m256 quantized_value = _mm256_fmadd_ps(ss, x_float, lb);

      // Load 8 floats from data and compute differences
      __m256 data_vec = _mm256_loadu_ps(&data[i]);
      __m256 dif      = _mm256_sub_ps(data_vec, quantized_value);

      // Compute squared differences and accumulate
      __m256 dif_sq = _mm256_mul_ps(dif, dif);
      sum_vec       = _mm256_add_ps(sum_vec, dif_sq);
    }

    // Handle remaining dimensions (if dim_ is not a multiple of 8)
    // float result = 0.0f;
    // for (; i < dim_; i++)
    // {
    //   const size_t index = i + data_index;
    //   const size_t group = index % dim_;
    //   const size_t pos = index * 4;
    //   const size_t byte_pos = pos / 8;
    //   const size_t is_high_nibble = i & 1;
    //   size_t combined = is_high_nibble ? (codes_[byte_pos] >> 4) & 0xF : codes_[byte_pos] & 0xF;
    //   const size_t offset = combined & 0x3;
    //   const size_t x = (combined >> 2) & 0x3;
    //   const size_t final_offset = offset + group * (1 << 2);
    //   const float quantized_value = lower_bound_[final_offset] + step_size_[final_offset] * static_cast<float>(x);
    //   float dif = data[i] - quantized_value;
    //   result += dif * dif;
    // }

    // Horizontal sum of the vectorized result
    __m256 hsum   = _mm256_hadd_ps(sum_vec, sum_vec);
    hsum          = _mm256_hadd_ps(hsum, hsum);
    __m128 low    = _mm256_castps256_ps128(hsum);
    __m128 high   = _mm256_extractf128_ps(hsum, 1);
    __m128 sum128 = _mm_add_ps(low, high);

    return _mm_cvtss_f32(sum128);
  }

  size_t size() const { return size_; }

  ~POUQ4() {
    delete[] lower_bound_;
    delete[] step_size_;
    delete[] codes_;
  }

private:
  size_t   size_        = 0;
  size_t   dim_         = 0;
  float   *lower_bound_ = nullptr;
  float   *step_size_   = nullptr;
  uint8_t *codes_       = nullptr;

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
};