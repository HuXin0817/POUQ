#pragma once

#include <immintrin.h>
#include <omp.h>
#include <tuple>

#include "clusterer.hpp"
#include "optimizer.hpp"

#include <assert.h>

namespace pouq {

inline void set(uint8_t *data, size_t index, size_t n) {
  const size_t byte_index = index >> 2;
  const size_t bit_offset = (index & 3) << 1;
  data[byte_index] &= ~(3 << bit_offset);
  data[byte_index] |= n << bit_offset;
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
  static inline __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);

public:
  explicit Quantizer(size_t groups) : dim_(groups) { assert(dim_ % 32 == 0); }

  void train(const float *data, size_t size) {
    std::vector<float> step_size(dim_ * 4);
    std::vector<float> lower_bound(dim_ * 4);

    combined_data_ = new std::tuple<uint8_t, uint8_t, uint16_t>[size / 4];

    std::vector<uint8_t> temp_cid(size / 4);
    std::vector<uint8_t> temp_code(size / 4);

    const size_t dim_div_4 = dim_ / 4;

#pragma omp parallel for
    for (size_t d = 0; d < dim_; d++) {
      const auto   data_freq_map = count_freq(data, size, d);
      const auto   bounds        = cluster(4, data_freq_map);
      const size_t d_times_4     = d * 4;

      for (size_t i = 0; i < bounds.size(); i++) {
        auto [lower, upper] = bounds[i];
        if (lower < upper) {
          auto data_start = std::lower_bound(data_freq_map.begin(),
              data_freq_map.end(),
              lower,
              [](const std::pair<float, size_t> &lhs, float rhs) -> bool { return lhs.first < rhs; });
          auto data_end   = std::upper_bound(data_freq_map.begin(),
              data_freq_map.end(),
              upper,
              [](float rhs, const std::pair<float, size_t> &lhs) -> bool { return rhs < lhs.first; });

          auto [opt_lower, opt_upper] = optimise(3, lower, upper, data_start, data_end);
          lower                       = opt_lower;
          upper                       = opt_upper;
        }
        lower_bound[d_times_4 + i] = lower;
        if (lower == upper) {
          step_size[d_times_4 + i] = 1.0;
        } else {
          step_size[d_times_4 + i] = (upper - lower) / 3.0f;
        }
      }

      for (size_t i = d; i < size; i += dim_) {
        const auto it = std::upper_bound(
            bounds.begin(), bounds.end(), data[i], [](float rhs, const std::pair<float, float> &lhs) -> bool {
              return rhs < lhs.first;
            });
        const size_t c = it - bounds.begin() - 1;
        const float  x =
            std::clamp((data[i] - lower_bound[d_times_4 + c]) / step_size[d_times_4 + c] + 0.5f, 0.0f, 3.0f);
        const size_t base_index = (i / dim_) * dim_div_4;
        set(&temp_cid[base_index], i % dim_, c);
        set(&temp_code[base_index], i % dim_, x);
      }
    }

    for (size_t i = 0; i < size / 4; i += 2) {
      uint16_t combined_code = static_cast<uint16_t>(temp_code[i + 1]) << 8 | temp_code[i];
      combined_data_[i / 2]  = std::make_tuple(temp_cid[i], temp_cid[i + 1], combined_code);
    }

    bounds_data_ = new std::pair<__m128, __m128>[dim_ / 4 * 256];

#pragma omp parallel for
    for (size_t g = 0; g < dim_ / 4; g++) {
      for (size_t j = 0; j < 256; j++) {
        auto [x0, x1, x2, x3] = get(j);
        const size_t base_idx = g * 16;

        __m128 lb = _mm_setr_ps(lower_bound[base_idx + 0 * 4 + x0],
            lower_bound[base_idx + 1 * 4 + x1],
            lower_bound[base_idx + 2 * 4 + x2],
            lower_bound[base_idx + 3 * 4 + x3]);
        __m128 st = _mm_setr_ps(step_size[base_idx + 0 * 4 + x0],
            step_size[base_idx + 1 * 4 + x1],
            step_size[base_idx + 2 * 4 + x2],
            step_size[base_idx + 3 * 4 + x3]);

        bounds_data_[g * 256 + j] = std::make_pair(lb, st);
      }
    }
  }

  float l2distance(const float *data, size_t offset) const {
    const size_t base_offset = offset / 4;
    __m256       sum8        = _mm256_setzero_ps();

    for (size_t i = 0; i < dim_; i += 8) {
      const size_t idx          = i / 4;
      const size_t combined_idx = (base_offset + idx) / 2;
      const auto [c1, c2, code] = combined_data_[combined_idx];
      const auto [lb1, st1]     = bounds_data_[idx * 256 + c1];
      const auto [lb2, st2]     = bounds_data_[(idx + 1) * 256 + c2];

      const __m256  lb_vec        = _mm256_insertf128_ps(_mm256_castps128_ps256(lb1), lb2, 1);
      const __m256  st_vec        = _mm256_insertf128_ps(_mm256_castps128_ps256(st1), st2, 1);
      const __m256i bytes         = _mm256_set1_epi32(code);
      const __m256i shifted       = _mm256_srlv_epi32(bytes, shifts);
      const __m256i masked        = _mm256_and_si256(shifted, _mm256_set1_epi32(3));
      const __m256  code_vec      = _mm256_cvtepi32_ps(masked);
      const __m256  reconstructed = _mm256_fmadd_ps(code_vec, st_vec, lb_vec);
      const __m256  data_vec      = _mm256_loadu_ps(data + i);
      const __m256  diff          = _mm256_sub_ps(reconstructed, data_vec);
      sum8                        = _mm256_add_ps(sum8, _mm256_mul_ps(diff, diff));
    }

    __m128 sum4 = _mm_add_ps(_mm256_extractf128_ps(sum8, 1), _mm256_castps256_ps128(sum8));
    sum4        = _mm_hadd_ps(sum4, sum4);
    sum4        = _mm_hadd_ps(sum4, sum4);

    return _mm_cvtss_f32(sum4);
  }

private:
  size_t                                  dim_           = 0;
  std::pair<__m128, __m128>              *bounds_data_   = nullptr;
  std::tuple<uint8_t, uint8_t, uint16_t> *combined_data_ = nullptr;

  std::vector<std::pair<float, size_t>> count_freq(const float *data, size_t size, size_t group) const {
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