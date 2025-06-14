#pragma once

#include <cstdint>
#include <utility>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define SIMD_ARM_NEON
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_AVX2
#endif

namespace pouq::simd {

#ifdef SIMD_ARM_NEON
inline float l2distance_simd_neon(const float *data,
    size_t                                     index,
    size_t                                     dim,
    const std::pair<float, float>             *code,
    const uint8_t                             *encode) {
  float32x4_t sum_vec = vdupq_n_f32(0.0f);
  size_t      i       = 0;

  for (; i + 3 < dim; i += 4) {
    float32x4_t data_vec = vld1q_f32(&data[i]);

    float v_array[4];
    for (int j = 0; j < 4; ++j) {
      const uint8_t c   = encode[index + i + j];
      const auto [l, s] = code[((c & 0xF) + (i + j) * 16)];
      v_array[j]        = l + s * (c >> 4 & 0xF);
    }

    float32x4_t v_vec = vld1q_f32(v_array);

    float32x4_t diff_vec = vsubq_f32(data_vec, v_vec);

    sum_vec = vfmaq_f32(sum_vec, diff_vec, diff_vec);
  }

  float dis = vaddvq_f32(sum_vec);

  for (; i < dim; ++i) {
    const uint8_t c   = encode[index + i];
    const auto [l, s] = code[((c & 0xF) + i * 16)];
    const float v     = l + s * (c >> 4 & 0xF);
    const float diff  = data[i] - v;
    dis += diff * diff;
  }

  return dis;
}
#endif

#ifdef SIMD_AVX2
inline float l2distance_simd_avx2(const float *data,
    size_t                                     index,
    size_t                                     dim,
    const std::pair<float, float>             *code,
    const uint8_t                             *encode) {
  __m256 sum_vec = _mm256_setzero_ps();
  size_t i       = 0;

  for (; i + 7 < dim; i += 8) {
    __m256 data_vec = _mm256_loadu_ps(&data[i]);

    float v_array[8];
    for (int j = 0; j < 8; ++j) {
      const uint8_t c   = encode[index + i + j];
      const auto [l, s] = code[((c & 0xF) + (i + j) * 16)];
      v_array[j]        = l + s * (c >> 4 & 0xF);
    }

    __m256 v_vec = _mm256_loadu_ps(v_array);

    __m256 diff_vec = _mm256_sub_ps(data_vec, v_vec);

    sum_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_vec);
  }

  float sum_array[8];
  _mm256_storeu_ps(sum_array, sum_vec);
  float dis = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum_array[4] + sum_array[5] + sum_array[6] +
              sum_array[7];

  for (; i < dim; ++i) {
    const uint8_t c   = encode[index + i];
    const auto [l, s] = code[((c & 0xF) + i * 16)];
    const float v     = l + s * (c >> 4 & 0xF);
    const float diff  = data[i] - v;
    dis += diff * diff;
  }

  return dis;
}
#endif

inline float l2distance_simd_generic(const float *data,
    size_t                                        index,
    size_t                                        dim,
    const std::pair<float, float>                *code,
    const uint8_t                                *encode) {
  float dis = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    const uint8_t c   = encode[index + i];
    const auto [l, s] = code[((c & 0xF) + i * 16)];
    const float v     = l + s * (c >> 4 & 0xF);
    const float diff  = data[i] - v;
    dis += diff * diff;
  }
  return dis;
}

inline float l2distance_simd(const float *data,
    size_t                                index,
    size_t                                dim,
    const std::pair<float, float>        *code,
    const uint8_t                        *encode) {
#ifdef SIMD_ARM_NEON
  return l2distance_simd_neon(data, index, dim, code, encode);
#elif defined(SIMD_AVX2)
  return l2distance_simd_avx2(data, index, dim, code, encode);
#else
  return l2distance_simd_generic(data, index, dim, code, encode);
#endif
}

}  // namespace pouq::simd