#pragma once

#include <omp.h>

#include "clusterer.hpp"
#include "optimizer.hpp"

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace pouq {

class POUQ8bit {
public:
  explicit POUQ8bit(size_t dim) : dim_(dim) {}

  void train(const float *data, size_t size) {
    codebook_ = new std::pair<float, float>[dim_ * (1 << 4)];
    codes_    = new uint8_t[size];

#pragma omp parallel for
    for (size_t group = 0; group < dim_; group++) {
      const auto data_freq_map = count_freq(data, size, group);
      const auto bounds        = clustering(1 << 4, data_freq_map);
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

          const auto [opt_lower, opt_upper] = optimizing(div, data_start, data_end);
          lower                             = opt_lower;
          upper                             = opt_upper;
        }
        if (lower == upper) {
          codebook_[offset + i] = {lower, 1.0};
        } else {
          codebook_[offset + i] = {lower, (upper - lower) / div};
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
        auto [lb, s]   = codebook_[offset + c];
        const float x  = std::clamp((d - lb) / s + 0.5f, 0.0f, div);
        set(codes_, 2 * i, c);
        set(codes_, 2 * i + 1, x);
      }
    }
  }

#if defined(__i386__) || defined(__x86_64__)
  float l2distance(const float *data, size_t n) const {
    float dis = 0.0f;

    const size_t simd_end = (dim_ / 8) * 8;
    __m256       sum_vec  = _mm256_setzero_ps();

    for (size_t i = 0; i < simd_end; i += 8) {

      __m256 data_vec = _mm256_loadu_ps(&data[i]);

      __m256 decoded_vec;
      {

        alignas(32) float   decoded[8];
        alignas(32) uint8_t codes_batch[8];

        for (size_t j = 0; j < 8; j++) {
          codes_batch[j] = codes_[n + i + j];
        }

        __m256i codes_vec   = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)codes_batch));
        __m256i lower_4bits = _mm256_and_si256(codes_vec, _mm256_set1_epi32(0xF));
        __m256i upper_4bits = _mm256_and_si256(_mm256_srli_epi32(codes_vec, 4), _mm256_set1_epi32(0xF));

        alignas(32) int32_t lower_indices[8];
        alignas(32) int32_t upper_values[8];
        _mm256_store_si256((__m256i *)lower_indices, lower_4bits);
        _mm256_store_si256((__m256i *)upper_values, upper_4bits);

        for (size_t j = 0; j < 8; j++) {
          const size_t codebook_idx = lower_indices[j] + (i + j) * (1 << 4);
          auto [lb, s]              = codebook_[codebook_idx];
          decoded[j]                = lb + s * static_cast<float>(upper_values[j]);
        }

        decoded_vec = _mm256_load_ps(decoded);
      }

      __m256 diff_vec = _mm256_sub_ps(data_vec, decoded_vec);

      sum_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_vec);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low  = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128  = _mm_add_ps(sum_high, sum_low);

    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    dis     = _mm_cvtss_f32(sum_128);

    for (size_t i = simd_end; i < dim_; i++) {
      uint8_t v    = codes_[n + i];
      auto [lb, s] = codebook_[((v & 0xF) + i * (1 << 4))];
      float decode = lb + s * (v >> 4 & 0xF);
      float diff   = data[i] - decode;
      dis += diff * diff;
    }

    return dis;
  }
#else

  float l2distance(const float *data, size_t n) const {
    float dis = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
      uint8_t v    = codes_[n + i];
      auto [lb, s] = codebook_[((v & 0xF) + i * (1 << 4))];
      float decode = lb + s * (v >> 4 & 0xF);
      float diff   = data[i] - decode;
      dis += diff * diff;
    }
    return dis;
  }

#endif

  ~POUQ8bit() {
    delete[] codebook_;
    delete[] codes_;
  }

private:
  size_t                   dim_      = 0;
  std::pair<float, float> *codebook_ = nullptr;
  uint8_t                 *codes_    = nullptr;

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

  void set(uint8_t *data, size_t index, size_t n) {
    n &= (1 << 4) - 1;
    const size_t byte_index = index / 2;
    const size_t nibble_pos = index % 2;

    if (nibble_pos == 0) {
      data[byte_index] = data[byte_index] & 0xF0 | n;
    } else {
      data[byte_index] = data[byte_index] & 0x0F | n << 4;
    }
  }
};

}  // namespace pouq