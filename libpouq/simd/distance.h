#pragma once

#include "def.h"

namespace pouq::simd {

#ifdef POUQ_X86_ARCH

static float
distance_avx2(int dim_, CodeUnit* code_, RecPara* rec_para_, const float* data, int offset) {
  assert(data != nullptr);
  assert(offset % dim_ == 0);

  __m256 sum_squares_vec = _mm256_setzero_ps();
  for (int dim = 0; dim < dim_; dim += 8) {
    int group_idx = dim / 4;
    auto [code1, code2, code_value] = code_.get()[(offset / 4 + group_idx) / 2];
    auto [lower1, step1] = rec_para_.get()[group_idx * 256 + code1];
    auto [lower2, step2] = rec_para_.get()[(group_idx + 1) * 256 + code2];

    __m256 lower_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(lower1), lower2, 1);
    __m256 step_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(step1), step2, 1);

    __m256i code_bytes = _mm256_set1_epi32(code_value);
    __m256i shift_amounts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    __m256i shifted_code = _mm256_srlv_epi32(code_bytes, shift_amounts);
    __m256i mask = _mm256_set1_epi32(3);
    __m256i masked_code = _mm256_and_si256(shifted_code, mask);

    __m256 code_vec = _mm256_cvtepi32_ps(masked_code);
    __m256 reconstructed_vec = _mm256_fmadd_ps(code_vec, step_vec, lower_vec);

    __m256 data_vec = _mm256_loadu_ps(data + dim);
    __m256 diff_vec = _mm256_sub_ps(reconstructed_vec, data_vec);
    sum_squares_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_squares_vec);
  }

  __m128 sum_low128 = _mm256_castps256_ps128(sum_squares_vec);
  __m128 sum_high128 = _mm256_extractf128_ps(sum_squares_vec, 1);
  __m128 total_sum128 = _mm_add_ps(sum_low128, sum_high128);

  __m128 shuffled_sum = _mm_movehdup_ps(total_sum128);
  total_sum128 = _mm_add_ps(total_sum128, shuffled_sum);
  shuffled_sum = _mm_movehl_ps(shuffled_sum, total_sum128);
  total_sum128 = _mm_add_ss(total_sum128, shuffled_sum);

  return _mm_cvtss_f32(total_sum128);
}

#endif

#if defined(POUQ_ARM_ARCH)

static float
distance_neon(
    int dim_, const CodeUnit* code_, const RecPara* rec_para_, const float* data, int offset) {
  float32x4_t sum_squares_vec = vdupq_n_f32(0.0f);
  for (int dim = 0; dim < dim_; dim += 8) {
    int group_idx = dim / 4;
    auto [code1, code2, code_value] = code_[(offset / 4 + group_idx) / 2];
    auto [lower1, step1] = rec_para_[group_idx * 256 + code1];
    auto [lower2, step2] = rec_para_[(group_idx + 1) * 256 + code2];

    uint32_t code_value_uint = code_value;

    uint32_t codes[8];
    codes[0] = (code_value_uint >> 0) & 3;
    codes[1] = (code_value_uint >> 2) & 3;
    codes[2] = (code_value_uint >> 4) & 3;
    codes[3] = (code_value_uint >> 6) & 3;
    codes[4] = (code_value_uint >> 8) & 3;
    codes[5] = (code_value_uint >> 10) & 3;
    codes[6] = (code_value_uint >> 12) & 3;
    codes[7] = (code_value_uint >> 14) & 3;

    uint32x4_t code_vec1 = vld1q_u32(&codes[0]);
    float32x4_t code_float1 = vcvtq_f32_u32(code_vec1);
    float32x4_t reconstructed_vec1 = vmlaq_f32(lower1, code_float1, step1);
    float32x4_t data_vec1 = vld1q_f32(data + dim);
    float32x4_t diff_vec1 = vsubq_f32(reconstructed_vec1, data_vec1);
    sum_squares_vec = vmlaq_f32(sum_squares_vec, diff_vec1, diff_vec1);

    uint32x4_t code_vec2 = vld1q_u32(&codes[4]);
    float32x4_t code_float2 = vcvtq_f32_u32(code_vec2);
    float32x4_t reconstructed_vec2 = vmlaq_f32(lower2, code_float2, step2);
    float32x4_t data_vec2 = vld1q_f32(data + dim + 4);
    float32x4_t diff_vec2 = vsubq_f32(reconstructed_vec2, data_vec2);
    sum_squares_vec = vmlaq_f32(sum_squares_vec, diff_vec2, diff_vec2);
  }

  float32x2_t sum_low = vadd_f32(vget_low_f32(sum_squares_vec), vget_high_f32(sum_squares_vec));
  float32x2_t sum_final = vpadd_f32(sum_low, sum_low);
  return vget_lane_f32(sum_final, 0);
}

#endif

static float
distance(int dim_, const CodeUnit* code_, const RecPara* rec_para_, const float* data, int offset) {
#ifdef POUQ_X86_ARCH
  return distance_avx2(dim_, code_, rec_para_, data, offset);
#elif defined(POUQ_ARM_ARCH)
  return distance_neon(dim_, code_, rec_para_, data, offset);
#endif
}

}  // namespace pouq::simd