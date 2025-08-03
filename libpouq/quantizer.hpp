#pragma once

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

#include "optimizer.hpp"
#include "quantizer.hpp"
#include "segmenter.hpp"

namespace pouq {

using CodeUnit = std::tuple<uint8_t, uint8_t, uint16_t>;
using RecPara = std::tuple<__m128, __m128>;

class Quantizer final {
  public:
  Quantizer() = default;

  explicit Quantizer(int dim) {
    init(dim);
  }

  void
  init(int dim) {
    dim_ = dim;
  }

  void
  train(const float* data, int size) {
    if (combined_data_) {
      _mm_free(combined_data_);
      combined_data_ = nullptr;
    }
    if (bounds_data_) {
      _mm_free(bounds_data_);
      bounds_data_ = nullptr;
    }

    int combined_data_size = size / 4;
    combined_data_ = static_cast<CodeUnit*>(_mm_malloc(combined_data_size * sizeof(CodeUnit), 256));
    if (!combined_data_) {
      throw std::bad_alloc();
    }

    int bounds_data_size = dim_ * 64;
    bounds_data_ = static_cast<RecPara*>(_mm_malloc(bounds_data_size * sizeof(RecPara), 256));
    if (!bounds_data_) {
      _mm_free(combined_data_);
      combined_data_ = nullptr;
      throw std::bad_alloc();
    }

    std::vector<float> step_size(dim_ * 4);
    std::vector<float> lower_bound(dim_ * 4);
    std::vector<uint8_t> cid(size / 4);
    std::vector<uint16_t> code(size / 8);

    const int dim_div_4 = dim_ / 4;

#pragma omp parallel for
    for (int d = 0; d < dim_; d++) {
      const auto data_freq_map = count_freq(data, size, d);
      const auto bounds = segment(4, data_freq_map);
      const int d_times_4 = d * 4;

      for (int i = 0; i < bounds.size(); i++) {
        auto [lower, upper] = bounds[i];
        if (lower < upper) {
          const auto data_start =
              std::lower_bound(data_freq_map.begin(),
                               data_freq_map.end(),
                               lower,
                               [](const std::pair<float, int>& lhs, const float rhs) -> bool {
                                 return lhs.first < rhs;
                               });
          const auto data_end =
              std::upper_bound(data_freq_map.begin(),
                               data_freq_map.end(),
                               upper,
                               [](const float rhs, const std::pair<float, int>& lhs) -> bool {
                                 return rhs < lhs.first;
                               });

          std::tie(lower, upper) = optimize(3, lower, upper, data_start, data_end);
        }
        lower_bound[d_times_4 + i] = lower;
        if (lower == upper) {
          step_size[d_times_4 + i] = 1.0;
        } else {
          step_size[d_times_4 + i] = (upper - lower) / 3.0f;
        }
      }

      for (int i = d; i < size; i += dim_) {
        const auto it = std::upper_bound(
            bounds.begin(),
            bounds.end(),
            data[i],
            [](float rhs, const std::pair<float, float>& lhs) -> bool { return rhs < lhs.first; });
        const int c = it - bounds.begin() - 1;
        const float x = std::clamp(
            (data[i] - lower_bound[d_times_4 + c]) / step_size[d_times_4 + c] + 0.5f, 0.0f, 3.0f);
        const int base_index = (i / dim_) * dim_div_4;
        set(&cid[base_index], i % dim_, c);
        set16(&code[base_index / 2], i % dim_, x);
      }
    }

    for (int i = 0; i < size / 4; i += 2) {
      combined_data_[i / 2] = std::make_tuple(cid[i], cid[i + 1], code[i / 2]);
    }

#pragma omp parallel for
    for (int g = 0; g < dim_ / 4; g++) {
      for (int j = 0; j < 256; j++) {
        const auto [x0, x1, x2, x3] = get(j);
        const int base_idx = g * 16;
        const __m128 lb = _mm_setr_ps(lower_bound[base_idx + 0 * 4 + x0],
                                      lower_bound[base_idx + 1 * 4 + x1],
                                      lower_bound[base_idx + 2 * 4 + x2],
                                      lower_bound[base_idx + 3 * 4 + x3]);
        const __m128 st = _mm_setr_ps(step_size[base_idx + 0 * 4 + x0],
                                      step_size[base_idx + 1 * 4 + x1],
                                      step_size[base_idx + 2 * 4 + x2],
                                      step_size[base_idx + 3 * 4 + x3]);
        bounds_data_[g * 256 + j] = {lb, st};
      }
    }
  }

  float
  l2distance(const float* data, int offset) const {
    offset /= 4;
    __m256 sum_vec = _mm256_setzero_ps();

    static const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    static const __m256i mask = _mm256_set1_epi32(3);

    for (int i = 0; i < dim_; i += 8) {
      const int idx = i / 4;
      const auto [c1, c2, code] = combined_data_[(offset + idx) / 2];
      const auto [lb1, st1] = bounds_data_[idx * 256 + c1];
      const auto [lb2, st2] = bounds_data_[(idx + 1) * 256 + c2];
      const __m256 lb_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(lb1), lb2, 1);
      const __m256 st_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(st1), st2, 1);
      const __m256i bytes = _mm256_set1_epi32(code);
      const __m256i shifted = _mm256_srlv_epi32(bytes, shifts);
      const __m256i masked = _mm256_and_si256(shifted, mask);
      const __m256 code_vec = _mm256_cvtepi32_ps(masked);
      const __m256 reconstructed = _mm256_fmadd_ps(code_vec, st_vec, lb_vec);
      const __m256 data_vec = _mm256_loadu_ps(data + i);
      const __m256 diff = _mm256_sub_ps(reconstructed, data_vec);
      sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    __m128 low128 = _mm256_castps256_ps128(sum_vec);
    __m128 high128 = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);
    __m128 shuf = _mm_movehdup_ps(sum128);
    sum128 = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sum128);
    sum128 = _mm_add_ss(sum128, shuf);
    return _mm_cvtss_f32(sum128);
  }

  ~Quantizer() {
    if (combined_data_) {
      _mm_free(combined_data_);
    }
    if (bounds_data_) {
      _mm_free(bounds_data_);
    }
  }

  private:
  int dim_ = 0;
  RecPara* bounds_data_ = nullptr;
  CodeUnit* combined_data_ = nullptr;

  std::vector<std::pair<float, int>>
  count_freq(const float* data, int size, int group) const {
    std::vector<float> sorted_data;
    sorted_data.reserve(size / dim_);
    for (int i = group; i < size; i += dim_) {
      sorted_data.push_back(data[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    float curr_value = sorted_data[0];
    int count = 1;
    std::vector<std::pair<float, int>> data_freq_map;
    data_freq_map.reserve(sorted_data.size());
    for (int i = 1; i < sorted_data.size(); i++) {
      if (sorted_data[i] == curr_value) {
        count++;
      } else {
        data_freq_map.emplace_back(curr_value, count);
        curr_value = sorted_data[i];
        count = 1;
      }
    }

    data_freq_map.emplace_back(curr_value, count);
    return data_freq_map;
  }

  static void
  set(uint8_t* data, int i, int n) {
    const int offset = (i & 3) << 1;
    i >>= 2;
    data[i] &= ~(3 << offset);
    data[i] |= n << offset;
  }

  static std::tuple<int, int, int, int>
  get(uint8_t byte) {
    return {
        byte & 3,
        byte >> 2 & 3,
        byte >> 4 & 3,
        byte >> 6 & 3,
    };
  }

  static void
  set16(uint16_t* data, int i, int n) {
    const int offset = (i & 7) << 1;
    i >>= 3;
    data[i] &= ~(3 << offset);
    data[i] |= n << offset;
  }
};

}  // namespace pouq