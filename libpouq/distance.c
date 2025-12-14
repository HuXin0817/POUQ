#include "distance.h"

__m256
cal_reconstructed_vec(int d, const CodeUnit* code, const RecPara* rec_para, int offset) {
  int group_idx = d / 4;
  CodeUnit code_unit = code[(offset / 4 + group_idx) / 2];
  RecPara rec_para1 = rec_para[group_idx * 256 + code_unit.x1];
  RecPara rec_para2 = rec_para[(group_idx + 1) * 256 + code_unit.x2];

  __m256 lower_vec =
      _mm256_insertf128_ps(_mm256_castps128_ps256(rec_para1.lower), rec_para2.lower, 1);
  __m256 step_vec =
      _mm256_insertf128_ps(_mm256_castps128_ps256(rec_para1.step_size), rec_para2.step_size, 1);

  __m256i code_bytes = _mm256_set1_epi32(code_unit.code);
  __m256i shift_amounts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
  __m256i shifted_code = _mm256_srlv_epi32(code_bytes, shift_amounts);
  __m256i mask = _mm256_set1_epi32(3);
  __m256i masked_code = _mm256_and_si256(shifted_code, mask);

  __m256 code_vec = _mm256_cvtepi32_ps(masked_code);
  return _mm256_fmadd_ps(code_vec, step_vec, lower_vec);
}

float
distance(int dim, const CodeUnit* code, const RecPara* rec_para, const float* data, int offset) {
  assert(data != NULL);
  assert(offset % dim == 0);

  __m256 sum_squares_vec = _mm256_setzero_ps();
  for (int d = 0; d < dim; d += 8) {
    __m256 reconstructed_vec = cal_reconstructed_vec(d, code, rec_para, offset);

    __m256 data_vec = _mm256_loadu_ps(data + d);
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

void
decode(int dim, const CodeUnit* code, const RecPara* rec_para, float* dist, int offset) {
  assert(dist != NULL);
  assert(offset % dim == 0);

  for (int d = 0; d < dim; d += 8) {
    __m256 reconstructed_vec = cal_reconstructed_vec(d, code, rec_para, offset);
    _mm256_storeu_ps(dist + d, reconstructed_vec);
  }
}

__m256
cal_reconstructed_vec_sq4(int d, const uint32_t* code, const SQ4RecPara* rec_para, int offset) {
  uint32_t code_value = code[(offset + d) / 8];

  SQ4RecPara para = rec_para[d / 8];
  __m256 lower_vec = para.lower;
  __m256 step_vec = para.step_size;

  __m256i code_int = _mm256_set1_epi32(code_value);
  __m256i shift_amounts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  __m256i shifted_code = _mm256_srlv_epi32(code_int, shift_amounts);
  __m256i mask = _mm256_set1_epi32(0xF);
  __m256i masked_code = _mm256_and_si256(shifted_code, mask);

  __m256 code_vec = _mm256_cvtepi32_ps(masked_code);

  return _mm256_fmadd_ps(code_vec, step_vec, lower_vec);
}

float
distance_sq4(
    int dim, const uint32_t* code, const SQ4RecPara* rec_para, const float* data, int offset) {
  assert(data != NULL);
  assert(offset % dim == 0);

  __m256 sum_squares_vec = _mm256_setzero_ps();
  for (int d = 0; d < dim; d += 8) {
    __m256 reconstructed_vec = cal_reconstructed_vec_sq4(d, code, rec_para, offset);

    __m256 data_vec = _mm256_loadu_ps(data + d);
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

void
decode_sq4(int dim, const uint32_t* code, const SQ4RecPara* rec_para, float* dist, int offset) {
  assert(dist != NULL);
  assert(offset % dim == 0);

  for (int d = 0; d < dim; d += 8) {
    __m256 reconstructed_vec = cal_reconstructed_vec_sq4(d, code, rec_para, offset);
    _mm256_storeu_ps(dist + d, reconstructed_vec);
  }
}
