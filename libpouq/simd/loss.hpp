#pragma once

#include "def.hpp"

namespace pouq::simd {

#include <cassert>
#include <cfloat>
#include <cmath>

#ifdef POUQ_X86_ARCH

template <bool do_count_freq>
static float
loss_avx2(
    float div, float lower, float step, const float* data_map, const int* freq_map, int size) {
  __m256 lower_vec = _mm256_set1_ps(lower);
  __m256 step_vec = _mm256_set1_ps(step);
  __m256 div_vec = _mm256_set1_ps(div);
  __m256 zero_vec = _mm256_setzero_ps();
  __m256 total_loss_vec = _mm256_setzero_ps();

  int simd_size = size - size % 8;
  for (int i = 0; i < simd_size; i += 8) {
    __m256 data_vec = _mm256_loadu_ps(&data_map[i]);
    __m256 real_quantized_code = _mm256_div_ps(_mm256_sub_ps(data_vec, lower_vec), step_vec);
    __m256 quantized_code = zero_vec;
    __m256 greater_mask = _mm256_cmp_ps(data_vec, lower_vec, _CMP_GT_OS);
    __m256 rounded_code =
        _mm256_round_ps(real_quantized_code, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    quantized_code = _mm256_blendv_ps(zero_vec, rounded_code, greater_mask);
    __m256 clamped_code = _mm256_min_ps(quantized_code, div_vec);
    quantized_code = _mm256_blendv_ps(zero_vec, clamped_code, greater_mask);
    __m256 code_loss = _mm256_sub_ps(real_quantized_code, quantized_code);
    __m256 loss_squared = _mm256_mul_ps(code_loss, code_loss);
    if (do_count_freq) {
      __m128i freq_low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&freq_map[i]));
      __m128i freq_high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&freq_map[i + 4]));
      __m256 freq_vec = _mm256_cvtepi32_ps(
          _mm256_inserti128_si256(_mm256_castsi128_si256(freq_low), freq_high, 1));
      loss_squared = _mm256_mul_ps(loss_squared, freq_vec);
    }
    total_loss_vec = _mm256_add_ps(total_loss_vec, loss_squared);
  }

  float total_loss = 0.0f;
  for (int i = simd_size; i < size; ++i) {
    float data_value = data_map[i];
    float real_quantized_code = (data_value - lower) / step;
    float quantized_code = 0.0f;

    if (data_value > lower) {
      quantized_code = std::round(real_quantized_code);
      if (quantized_code > div) {
        quantized_code = div;
      }
    }

    float code_loss = real_quantized_code - quantized_code;
    if (do_count_freq) {
      total_loss += code_loss * code_loss * static_cast<float>(freq_map[i]);
    } else {
      total_loss += code_loss * code_loss;
    }
  }

  __m128 sum_low = _mm256_castps256_ps128(total_loss_vec);
  __m128 sum_high = _mm256_extractf128_ps(total_loss_vec, 1);
  __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
  __m128 shuffled = _mm_movehdup_ps(sum_128);
  sum_128 = _mm_add_ps(sum_128, shuffled);
  shuffled = _mm_movehl_ps(shuffled, sum_128);
  sum_128 = _mm_add_ss(sum_128, shuffled);
  total_loss += _mm_cvtss_f32(sum_128);

  return total_loss * step * step;
}

#endif

#ifdef POUQ_ARM_ARCH

template <bool do_count_freq>
static float
loss_neon(
    float div, float lower, float step, const float* data_map, const int* freq_map, int size) {
  float32x4_t lower_vec = vdupq_n_f32(lower);
  float32x4_t step_vec = vdupq_n_f32(step);
  float32x4_t div_vec = vdupq_n_f32(div);
  float32x4_t zero_vec = vdupq_n_f32(0.0f);
  float32x4_t total_loss_vec = vdupq_n_f32(0.0f);

  int simd_size = size - size % 4;
  for (int i = 0; i < simd_size; i += 4) {
    float32x4_t data_vec = vld1q_f32(&data_map[i]);
    float32x4_t real_quantized_code = vdivq_f32(vsubq_f32(data_vec, lower_vec), step_vec);
    float32x4_t quantized_code = zero_vec;
    uint32x4_t greater_mask = vcgtq_f32(data_vec, lower_vec);
    float32x4_t rounded_code = vaddq_f32(real_quantized_code, vdupq_n_f32(0.5f));
    int32x4_t rounded_int = vcvtq_s32_f32(rounded_code);
    float32x4_t rounded_float = vcvtq_f32_s32(rounded_int);
    quantized_code = vbslq_f32(greater_mask, rounded_float, zero_vec);
    float32x4_t clamped_code = vminq_f32(quantized_code, div_vec);
    quantized_code = vbslq_f32(greater_mask, clamped_code, zero_vec);
    float32x4_t code_loss = vsubq_f32(real_quantized_code, quantized_code);
    float32x4_t loss_squared = vmulq_f32(code_loss, code_loss);
    if (do_count_freq) {
      int32x4_t freq_vec = vld1q_s32(&freq_map[i]);
      float32x4_t freq_float = vcvtq_f32_s32(freq_vec);
      loss_squared = vmulq_f32(loss_squared, freq_float);
    }
    total_loss_vec = vaddq_f32(total_loss_vec, loss_squared);
  }

  float total_loss = 0.0f;
  for (int i = simd_size; i < size; ++i) {
    float data_value = data_map[i];
    float real_quantized_code = (data_value - lower) / step;
    float quantized_code = 0.0f;

    if (data_value > lower) {
      quantized_code = std::round(real_quantized_code);
      if (quantized_code > div) {
        quantized_code = div;
      }
    }

    float code_loss = real_quantized_code - quantized_code;
    if (do_count_freq) {
      total_loss += code_loss * code_loss * static_cast<float>(freq_map[i]);
    } else {
      total_loss += code_loss * code_loss;
    }
  }

  float32x2_t sum_low = vadd_f32(vget_low_f32(total_loss_vec), vget_high_f32(total_loss_vec));
  float32x2_t sum_final = vpadd_f32(sum_low, sum_low);
  total_loss += vget_lane_f32(sum_final, 0);

  return total_loss * step * step;
}

#endif

template <bool do_count_freq>
static float
loss(float div, float lower, float step, const float* data_map, const int* freq_map, int size) {
  assert(div > 0.0f);
  assert(step >= FLT_EPSILON);
  assert(size > 0);
  assert(data_map != nullptr);
  if (do_count_freq) {
    assert(freq_map != nullptr);
  }

#ifdef POUQ_X86_ARCH
  return loss_avx2<do_count_freq>(div, lower, step, data_map, freq_map, size);
#elif defined(POUQ_ARM_ARCH)
  return loss_neon<do_count_freq>(div, lower, step, data_map, freq_map, size);
#endif
}

}  // namespace pouq::simd