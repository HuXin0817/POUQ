#pragma once

#include <vector>

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

inline float calculate_quantization_loss(const float             division_count,
    float                                                        cluster_lower_bound,
    float                                                        step_size,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_end) {
  step_size        = std::max(step_size, 1e-8f);
  float total_loss = 0.0f;

  auto         it             = data_begin;
  const size_t total_elements = std::distance(data_begin, data_end);

#if defined(__AVX2__) && (defined(__i386__) || defined(__x86_64__))
  const __m256 step_size_vec           = _mm256_set1_ps(step_size);
  const __m256 cluster_lower_bound_vec = _mm256_set1_ps(cluster_lower_bound);
  const __m256 division_count_vec      = _mm256_set1_ps(division_count);
  const __m256 zero_vec                = _mm256_setzero_ps();
  const __m256 one_vec                 = _mm256_set1_ps(1.0f);
  __m256       total_loss_vec          = _mm256_setzero_ps();

  const size_t simd_end = (total_elements / 8) * 8;
  for (size_t i = 0; i < simd_end && it != data_end; i += 8) {
    alignas(32) float data_values[8];
    alignas(32) float point_counts[8];

    for (int j = 0; j < 8 && it != data_end; ++j, ++it) {
      data_values[j]  = it->first;
      point_counts[j] = static_cast<float>(it->second);
    }

    const __m256 data_vec  = _mm256_load_ps(data_values);
    const __m256 count_vec = _mm256_load_ps(point_counts);

    const __m256 diff_vec                = _mm256_sub_ps(data_vec, cluster_lower_bound_vec);
    const __m256 real_quantized_code_vec = _mm256_div_ps(diff_vec, step_size_vec);

    __m256       quantized_code_vec = _mm256_setzero_ps();
    const __m256 mask_positive      = _mm256_cmp_ps(data_vec, cluster_lower_bound_vec, _CMP_GT_OQ);

    __m256 rounded_vec = _mm256_round_ps(real_quantized_code_vec, _MM_FROUND_TO_NEAREST_INT);
    rounded_vec        = _mm256_min_ps(rounded_vec, division_count_vec);
    quantized_code_vec = _mm256_and_ps(rounded_vec, mask_positive);

    const __m256 error_vec          = _mm256_sub_ps(real_quantized_code_vec, quantized_code_vec);
    const __m256 error_squared_vec  = _mm256_mul_ps(error_vec, error_vec);
    const __m256 weighted_error_vec = _mm256_mul_ps(error_squared_vec, count_vec);

    total_loss_vec = _mm256_add_ps(total_loss_vec, weighted_error_vec);
  }

  alignas(32) float loss_array[8];
  _mm256_store_ps(loss_array, total_loss_vec);
  for (int i = 0; i < 8; ++i) {
    total_loss += loss_array[i];
  }

#elif defined(__ARM_NEON) || defined(__aarch64__)
  const float32x4_t step_size_vec           = vdupq_n_f32(step_size);
  const float32x4_t cluster_lower_bound_vec = vdupq_n_f32(cluster_lower_bound);
  const float32x4_t division_count_vec      = vdupq_n_f32(division_count);
  const float32x4_t zero_vec                = vdupq_n_f32(0.0f);
  float32x4_t       total_loss_vec          = vdupq_n_f32(0.0f);

  const size_t simd_end = (total_elements / 4) * 4;
  for (size_t i = 0; i < simd_end && it != data_end; i += 4) {
    float data_values[4];
    float point_counts[4];

    for (int j = 0; j < 4 && it != data_end; ++j, ++it) {
      data_values[j]  = it->first;
      point_counts[j] = static_cast<float>(it->second);
    }

    const float32x4_t data_vec  = vld1q_f32(data_values);
    const float32x4_t count_vec = vld1q_f32(point_counts);

    const float32x4_t diff_vec                = vsubq_f32(data_vec, cluster_lower_bound_vec);
    const float32x4_t real_quantized_code_vec = vdivq_f32(diff_vec, step_size_vec);

    float32x4_t      quantized_code_vec = vdupq_n_f32(0.0f);
    const uint32x4_t mask_positive      = vcgtq_f32(data_vec, cluster_lower_bound_vec);

    float32x4_t rounded_vec = vrndnq_f32(real_quantized_code_vec);
    rounded_vec             = vminq_f32(rounded_vec, division_count_vec);
    quantized_code_vec      = vbslq_f32(mask_positive, rounded_vec, zero_vec);

    const float32x4_t error_vec          = vsubq_f32(real_quantized_code_vec, quantized_code_vec);
    const float32x4_t error_squared_vec  = vmulq_f32(error_vec, error_vec);
    const float32x4_t weighted_error_vec = vmulq_f32(error_squared_vec, count_vec);

    total_loss_vec = vaddq_f32(total_loss_vec, weighted_error_vec);
  }

  float32x2_t sum_pair  = vadd_f32(vget_low_f32(total_loss_vec), vget_high_f32(total_loss_vec));
  float32x2_t sum_final = vpadd_f32(sum_pair, sum_pair);
  total_loss += vget_lane_f32(sum_final, 0);

#endif

  for (; it != data_end; ++it) {
    const auto &[data_value, point_count] = *it;
    const float real_quantized_code       = (data_value - cluster_lower_bound) / step_size;
    float       quantized_code            = 0.0f;

    if (data_value > cluster_lower_bound) {
      quantized_code = std::round(real_quantized_code);
      if (quantized_code > division_count) {
        quantized_code = division_count;
      }
    }

    const float quantization_error = real_quantized_code - quantized_code;
    total_loss += quantization_error * quantization_error * static_cast<float>(point_count);
  }

  return total_loss * step_size * step_size;
}
