#pragma once

#include <immintrin.h>
#include <omp.h>

#include "clusterer.hpp"
#include "optimizer.hpp"

#include <assert.h>

namespace pouq {

inline void set(uint8_t *data, size_t index, size_t n) {
  n &= 3;
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

public:
  explicit Quantizer(size_t groups) : dim_(groups) { assert(dim_ % 32 == 0); }

  void train(const float *data, size_t size) {
    std::vector<float> step_size(dim_ * 4);
    std::vector<float> lower_bound(dim_ * 4);
    cid_  = new uint8_t[size / 4];
    code_ = new uint8_t[size / 4];

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
        set(&cid_[base_index], i % dim_, c);
        set(&code_[base_index], i % dim_, x);
      }
    }

    lower_bound_ = new float[dim_ * 256];
    step_size_   = new float[dim_ * 256];

#pragma omp parallel for
    for (size_t i = 0; i < dim_; i += 4) {
      const size_t base_idx = (i / 4) * 1024;
      for (size_t j = 0; j < 256; j++) {
        auto [x0, x1, x2, x3] = get(j);
        const size_t offset   = base_idx + j * 4;

        lower_bound_[offset + 0] = lower_bound[(i + 0) * 4 + x0];
        lower_bound_[offset + 1] = lower_bound[(i + 1) * 4 + x1];
        lower_bound_[offset + 2] = lower_bound[(i + 2) * 4 + x2];
        lower_bound_[offset + 3] = lower_bound[(i + 3) * 4 + x3];

        step_size_[offset + 0] = step_size[(i + 0) * 4 + x0];
        step_size_[offset + 1] = step_size[(i + 1) * 4 + x1];
        step_size_[offset + 2] = step_size[(i + 2) * 4 + x2];
        step_size_[offset + 3] = step_size[(i + 3) * 4 + x3];
      }
    }
  }

  float l2distance(const float *data, size_t offset) const {
    const size_t base_offset = offset / 4;
    __m256       sum8        = _mm256_setzero_ps();

    for (size_t i = 0; i < dim_; i += 8) {

      const size_t group_idx1 = i / 4;
      const size_t group_idx2 = (i + 4) / 4;

      const uint8_t cid_byte1  = cid_[base_offset + group_idx1];
      const uint8_t cid_byte2  = cid_[base_offset + group_idx2];
      const uint8_t code_byte1 = code_[base_offset + group_idx1];
      const uint8_t code_byte2 = code_[base_offset + group_idx2];

      const size_t lookup_base1 = group_idx1 * 1024 + cid_byte1 * 4;
      const size_t lookup_base2 = group_idx2 * 1024 + cid_byte2 * 4;

      __m128 step1  = _mm_loadu_ps(step_size_ + lookup_base1);
      __m128 step2  = _mm_loadu_ps(step_size_ + lookup_base2);
      __m128 lower1 = _mm_loadu_ps(lower_bound_ + lookup_base1);
      __m128 lower2 = _mm_loadu_ps(lower_bound_ + lookup_base2);

      __m256 step_vec  = _mm256_set_m128(step2, step1);
      __m256 lower_vec = _mm256_set_m128(lower2, lower1);

      __m256 code_vec = _mm256_setr_ps((code_byte1 >> 0) & 3,
          (code_byte1 >> 2) & 3,
          (code_byte1 >> 4) & 3,
          (code_byte1 >> 6) & 3,
          (code_byte2 >> 0) & 3,
          (code_byte2 >> 2) & 3,
          (code_byte2 >> 4) & 3,
          (code_byte2 >> 6) & 3);

      __m256 reconstructed = _mm256_fmadd_ps(code_vec, step_vec, lower_vec);

      __m256 data_vec = _mm256_loadu_ps(data + i);

      __m256 diff = _mm256_sub_ps(reconstructed, data_vec);

      sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(diff, diff));
    }

    __m128 sum4 = _mm_add_ps(_mm256_extractf128_ps(sum8, 1), _mm256_castps256_ps128(sum8));
    sum4        = _mm_hadd_ps(sum4, sum4);
    sum4        = _mm_hadd_ps(sum4, sum4);

    return _mm_cvtss_f32(sum4);
  }

private:
  size_t   dim_ = 0;
  float   *lower_bound_;
  float   *step_size_;
  uint8_t *cid_;
  uint8_t *code_;

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