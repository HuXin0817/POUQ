#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define SIMD_ARM_NEON
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_AVX2
#endif

namespace pouq::simd {

#ifdef SIMD_ARM_NEON
inline float quantization_loss_simd_neon(const float             d,
    float                                                        l,
    float                                                        s,
    const std::vector<std::pair<float, size_t>>::const_iterator &begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &end) {
  s = std::max(s, 1e-8f);

  const float32x4_t d_vec    = vdupq_n_f32(d);
  const float32x4_t l_vec    = vdupq_n_f32(l);
  const float32x4_t s_vec    = vdupq_n_f32(s);
  const float32x4_t zero_vec = vdupq_n_f32(0.0f);
  const float32x4_t half_vec = vdupq_n_f32(0.5f);

  float32x4_t loss_vec = vdupq_n_f32(0.0f);

  auto it = begin;

  while (std::distance(it, end) >= 4) {
    float v_array[4], c_array[4];
    for (int i = 0; i < 4; ++i) {
      v_array[i] = it->first;
      c_array[i] = static_cast<float>(it->second);
      ++it;
    }

    float32x4_t v_vec = vld1q_f32(v_array);
    float32x4_t c_vec = vld1q_f32(c_array);

    float32x4_t rc_vec = vdivq_f32(vsubq_f32(v_vec, l_vec), s_vec);

    float32x4_t qc_vec = zero_vec;

    uint32x4_t v_gt_l_mask = vcgtq_f32(v_vec, l_vec);

    float32x4_t rc_rounded = vaddq_f32(rc_vec, half_vec);
    rc_rounded             = vcvtq_f32_s32(vcvtq_s32_f32(rc_rounded));

    rc_rounded = vmaxq_f32(rc_rounded, zero_vec);
    rc_rounded = vminq_f32(rc_rounded, d_vec);

    qc_vec = vbslq_f32(v_gt_l_mask, rc_rounded, zero_vec);

    float32x4_t err_vec = vsubq_f32(rc_vec, qc_vec);

    float32x4_t err_squared  = vmulq_f32(err_vec, err_vec);
    float32x4_t weighted_err = vmulq_f32(err_squared, c_vec);

    loss_vec = vaddq_f32(loss_vec, weighted_err);
  }

  float loss = vaddvq_f32(loss_vec);

  for (; it != end; ++it) {
    const auto &[v, c] = *it;
    const float rc     = (v - l) / s;
    float       qc     = 0.0f;
    if (v > l) {
      qc = std::round(rc);
      if (qc > d) {
        qc = d;
      }
    }
    const float err = rc - qc;
    loss += err * err * static_cast<float>(c);
  }

  return loss * s * s;
}
#endif

#ifdef SIMD_AVX2
inline float quantization_loss_simd_avx2(const float             d,
    float                                                        l,
    float                                                        s,
    const std::vector<std::pair<float, size_t>>::const_iterator &begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &end) {
  s = std::max(s, 1e-8f);

  const __m256 d_vec    = _mm256_set1_ps(d);
  const __m256 l_vec    = _mm256_set1_ps(l);
  const __m256 s_vec    = _mm256_set1_ps(s);
  const __m256 zero_vec = _mm256_setzero_ps();
  const __m256 half_vec = _mm256_set1_ps(0.5f);

  __m256 loss_vec = _mm256_setzero_ps();

  auto it = begin;

  while (std::distance(it, end) >= 8) {
    float v_array[8], c_array[8];
    for (int i = 0; i < 8; ++i) {
      v_array[i] = it->first;
      c_array[i] = static_cast<float>(it->second);
      ++it;
    }

    __m256 v_vec = _mm256_loadu_ps(v_array);
    __m256 c_vec = _mm256_loadu_ps(c_array);

    __m256 rc_vec = _mm256_div_ps(_mm256_sub_ps(v_vec, l_vec), s_vec);

    __m256 qc_vec = zero_vec;

    __m256 v_gt_l_mask = _mm256_cmp_ps(v_vec, l_vec, _CMP_GT_OQ);

    __m256 rc_rounded = _mm256_add_ps(rc_vec, half_vec);
    rc_rounded        = _mm256_cvtepi32_ps(_mm256_cvtps_epi32(rc_rounded));

    rc_rounded = _mm256_max_ps(rc_rounded, zero_vec);
    rc_rounded = _mm256_min_ps(rc_rounded, d_vec);

    qc_vec = _mm256_blendv_ps(zero_vec, rc_rounded, v_gt_l_mask);

    __m256 err_vec = _mm256_sub_ps(rc_vec, qc_vec);

    __m256 err_squared  = _mm256_mul_ps(err_vec, err_vec);
    __m256 weighted_err = _mm256_mul_ps(err_squared, c_vec);

    loss_vec = _mm256_add_ps(loss_vec, weighted_err);
  }

  float sum_array[8];
  _mm256_storeu_ps(sum_array, loss_vec);
  float loss = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum_array[4] + sum_array[5] + sum_array[6] +
               sum_array[7];

  for (; it != end; ++it) {
    const auto &[v, c] = *it;
    const float rc     = (v - l) / s;
    float       qc     = 0.0f;
    if (v > l) {
      qc = std::round(rc);
      if (qc > d) {
        qc = d;
      }
    }
    const float err = rc - qc;
    loss += err * err * static_cast<float>(c);
  }

  return loss * s * s;
}
#endif

inline float quantization_loss_simd_generic(const float          d,
    float                                                        l,
    float                                                        s,
    const std::vector<std::pair<float, size_t>>::const_iterator &begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &end) {
  s          = std::max(s, 1e-8f);
  float loss = 0.0f;

  for (auto it = begin; it != end; ++it) {
    const auto &[v, c] = *it;
    const float rc     = (v - l) / s;
    float       qc     = 0.0f;
    if (v > l) {
      qc = std::round(rc);
      if (qc > d) {
        qc = d;
      }
    }
    const float err = rc - qc;
    loss += err * err * static_cast<float>(c);
  }

  return loss * s * s;
}

inline float quantization_loss_simd(const float                  d,
    float                                                        l,
    float                                                        s,
    const std::vector<std::pair<float, size_t>>::const_iterator &begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &end) {
#ifdef SIMD_ARM_NEON
  return quantization_loss_simd_neon(d, l, s, begin, end);
#elif defined(SIMD_AVX2)
  return quantization_loss_simd_avx2(d, l, s, begin, end);
#else
  return quantization_loss_simd_generic(d, l, s, begin, end);
#endif
}

}  // namespace pouq::simd