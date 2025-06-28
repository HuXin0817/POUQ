#pragma once

#include <omp.h>

#include "clusterer.hpp"
#include "optimizer.hpp"

#include <assert.h>

#include <immintrin.h>

namespace pouq {

inline void set(uint8_t *data, size_t index, size_t n) {
  n &= 3;
  const size_t byte_index = index >> 2;        // 等价于 index * 2 / 8，但更高效
  const size_t bit_offset = (index & 3) << 1;  // 等价于 (index * 2) % 8，但更高效
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
    std::vector<float> step_size_(dim_ * 4);
    std::vector<float> lower_bound_(dim_ * 4);
    cid_  = new uint8_t[size / 4];
    code_ = new uint8_t[size / 4];

    const size_t dim_div_4 = dim_ / 4;  // 预计算常用值

#pragma omp parallel for
    for (size_t d = 0; d < dim_; d++) {
      const auto   data_freq_map = count_freq(data, size, d);
      const auto   bounds        = cluster(4, data_freq_map);
      const size_t d_times_4     = d * 4;  // 预计算 d * 4

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
        lower_bound_[d_times_4 + i] = lower;
        if (lower == upper) {
          step_size_[d_times_4 + i] = 1.0;
        } else {
          step_size_[d_times_4 + i] = (upper - lower) / 3.0f;
        }
      }

      for (size_t i = d; i < size; i += dim_) {
        const auto it = std::upper_bound(
            bounds.begin(), bounds.end(), data[i], [](const float rhs, const std::pair<float, float> &lhs) -> bool {
              return rhs < lhs.first;
            });
        const size_t c = it - bounds.begin() - 1;
        const float  x =
            std::clamp((data[i] - lower_bound_[d_times_4 + c]) / step_size_[d_times_4 + c] + 0.5f, 0.0f, 3.0f);
        const size_t base_index = (i / dim_) * dim_div_4;  // 预计算基础索引
        set(&cid_[base_index], i % dim_, c);
        set(&code_[base_index], i % dim_, x);
      }
    }

    lower_bound_128 = new float[dim_ * 256];
    step_size_128   = new float[dim_ * 256];

#pragma omp parallel for
    for (size_t i = 0; i < dim_; i += 4) {
      for (size_t j = 0; j < 256; j++) {
        const size_t base_idx = (i / 4) * 256 * 8 + j * 8;  // 每组连续存储8个float
        auto [x0, x1, x2, x3] = get(j);

        lower_bound_128[base_idx + 0] = lower_bound_[(i + 0) * 4 + x0];
        lower_bound_128[base_idx + 1] = step_size_[(i + 0) * 4 + x0];
        lower_bound_128[base_idx + 2] = lower_bound_[(i + 1) * 4 + x1];
        lower_bound_128[base_idx + 3] = step_size_[(i + 1) * 4 + x1];
        lower_bound_128[base_idx + 4] = lower_bound_[(i + 2) * 4 + x2];
        lower_bound_128[base_idx + 5] = step_size_[(i + 2) * 4 + x2];
        lower_bound_128[base_idx + 6] = lower_bound_[(i + 3) * 4 + x3];
        lower_bound_128[base_idx + 7] = step_size_[(i + 3) * 4 + x3];
      }
    }
  }

  float l2distance(const float *data, size_t offset) const {
    const size_t base_offset = offset / 4;
    __m256       acc         = _mm256_setzero_ps();

    for (size_t i = 0; i < dim_; i += 8) {
      // 加载cid和code
      const uint8_t *cid_ptr   = &cid_[base_offset + i / 4];
      const uint8_t *code_ptr  = &code_[base_offset + i / 4];
      const uint16_t cid_pair  = *reinterpret_cast<const uint16_t *>(cid_ptr);
      const uint16_t code_pair = *reinterpret_cast<const uint16_t *>(code_ptr);

      // 计算查表地址
      const size_t lookup_idx0 = (i / 4) * 256 * 8 + (cid_pair & 0xFF) * 8;
      const size_t lookup_idx1 = (i / 4) * 256 * 8 + (cid_pair >> 8) * 8;

      // 加载下界和步长 (8个连续float)
      __m256 lb_step0 = _mm256_loadu_ps(lower_bound_128 + lookup_idx0);
      __m256 lb_step1 = _mm256_loadu_ps(lower_bound_128 + lookup_idx1);

      // 重组数据：低128位=前4维，高128位=后4维
      __m256 lb   = _mm256_set_m128(_mm256_extractf128_ps(lb_step1, 0), _mm256_extractf128_ps(lb_step0, 0));
      __m256 step = _mm256_set_m128(_mm256_extractf128_ps(lb_step1, 1), _mm256_extractf128_ps(lb_step0, 1));

      // 提取量化值 (使用SSE优化)
      __m128i code_vec = _mm_set1_epi16(code_pair);
      __m128i quant    = _mm_srlv_epi16(code_vec, _mm_set_epi32(0, 2, 4, 6, 8, 10, 12, 14));
      quant            = _mm_and_si128(quant, _mm_set1_epi8(0x03));
      __m256 quant_f   = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(quant));

      // 重建并计算平方差
      __m256 recon    = _mm256_fmadd_ps(quant_f, step, lb);
      __m256 data_vec = _mm256_loadu_ps(data + offset + i);
      __m256 diff     = _mm256_sub_ps(recon, data_vec);
      acc             = _mm256_fmadd_ps(diff, diff, acc);
    }

    // 水平求和
    __m128 sum4 = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    sum4        = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(_mm_hadd_ps(sum4, sum4));
  }

private:
  size_t   dim_ = 0;
  float   *lower_bound_128;
  float   *step_size_128;
  uint8_t *cid_;
  uint8_t *code_;

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