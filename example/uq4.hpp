#pragma once

#include <algorithm>
#include <cassert>
#include <immintrin.h>
#include <omp.h>
#include <vector>

class UQQuantizer {
public:
  explicit UQQuantizer(size_t groups) : dim_(groups), num_vectors_(dim_ / 8) {
    assert(dim_ % 32 == 0);
    lowers_     = static_cast<__m256 *>(_mm_malloc(num_vectors_ * sizeof(__m256), 32));
    step_sizes_ = static_cast<__m256 *>(_mm_malloc(num_vectors_ * sizeof(__m256), 32));
  }

  ~UQQuantizer() {
    delete[] code;
    _mm_free(lowers_);
    _mm_free(step_sizes_);
  }

  void train(const float *data, size_t size) {
    assert(size % dim_ == 0);
    size_t n_samples = size / dim_;

    std::vector<float> lowers(dim_);
    std::vector<float> step_sizes(dim_);

    for (size_t d = 0; d < dim_; d++) {
      float min_val = data[d];
      float max_val = data[d];
      for (size_t i = 1; i < n_samples; i++) {
        float val = data[i * dim_ + d];
        min_val   = std::min(min_val, val);
        max_val   = std::max(max_val, val);
      }
      lowers[d]     = min_val;
      step_sizes[d] = (min_val == max_val) ? 1.0f : (max_val - min_val) / 15.0f;
    }

    for (size_t i = 0; i < num_vectors_; i++) {
      float lower_array[8];
      float step_array[8];
      for (size_t j = 0; j < 8; j++) {
        size_t idx     = i * 8 + j;
        lower_array[j] = lowers[idx];
        step_array[j]  = step_sizes[idx];
      }
      lowers_[i]     = _mm256_loadu_ps(lower_array);
      step_sizes_[i] = _mm256_loadu_ps(step_array);
    }

    code = new uint32_t[size / 8];
    for (size_t i = 0; i < size; i += 8) {
      uint32_t packed = 0;
      for (int j = 0; j < 8 && i + j < size; j++) {
        size_t   idx = i + j;
        size_t   d   = idx % dim_;
        float    val = data[idx];
        uint32_t q   = std::min<uint32_t>(15, std::max<uint32_t>(0, std::round((val - lowers[d]) / step_sizes[d])));
        packed |= (q << ((7 - j) * 4));
      }
      code[i / 8] = packed;
    }
  }

  float l2distance(const float *data, size_t offset) const {
    offset /= 8;
    __m256 sum_vec = _mm256_setzero_ps();

    for (size_t i = 0; i < num_vectors_; i++) {
      const uint32_t       packed        = code[offset + i];
      const __m256i        packed_vec    = _mm256_set1_epi32(packed);
      static const __m256i shift_amounts = _mm256_setr_epi32(28, 24, 20, 16, 12, 8, 4, 0);
      static const __m256i mask          = _mm256_set1_epi32(0xF);
      __m256i              shifted       = _mm256_srlv_epi32(packed_vec, shift_amounts);
      __m256i              q_vals_vec    = _mm256_and_si256(shifted, mask);
      __m256               q_float       = _mm256_cvtepi32_ps(q_vals_vec);

      __m256 lower_vec     = lowers_[i];
      __m256 step_size_vec = step_sizes_[i];
      __m256 data_vec      = _mm256_loadu_ps(&data[i * 8]);

      __m256 decoded = _mm256_fmadd_ps(step_size_vec, q_float, lower_vec);
      __m256 diff    = _mm256_sub_ps(decoded, data_vec);
      sum_vec        = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    __m128 low128  = _mm256_castps256_ps128(sum_vec);
    __m128 high128 = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128  = _mm_add_ps(low128, high128);
    __m128 shuf    = _mm_movehdup_ps(sum128);
    sum128         = _mm_add_ps(sum128, shuf);
    shuf           = _mm_movehl_ps(shuf, sum128);
    sum128         = _mm_add_ss(sum128, shuf);
    return _mm_cvtss_f32(sum128);
  }

  void l2distance_batch(const float *data, size_t size, float *distance) const {
#pragma omp parallel for
    for (size_t i = 0; i < size; i += dim_) {
      distance[i / dim_] = l2distance(data, i);
    }
  }

private:
  size_t    dim_         = 0;
  size_t    num_vectors_ = 0;
  __m256   *lowers_      = nullptr;
  __m256   *step_sizes_  = nullptr;
  uint32_t *code         = nullptr;
};