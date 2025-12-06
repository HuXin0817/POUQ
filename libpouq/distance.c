#include "distance.h"

float
distance(int dim, const CodeUnit* code, const RecPara* rec_para, const float* data, int offset) {
  assert(data != NULL);
  assert(offset % dim == 0);

  __m256 sum_squares_vec = _mm256_setzero_ps();
  for (int d = 0; d < dim; d += 8) {
    int group_idx = d / 4;
    CodeUnit code_unit = code[(offset / 4 + group_idx) / 2];
    uint8_t code1 = code_unit.x1;
    uint8_t code2 = code_unit.x2;
    uint16_t code_value = code_unit.code;
    RecPara rec_para1 = rec_para[group_idx * 256 + code1];
    __m128 lower1 = rec_para1.lower;
    __m128 step1 = rec_para1.step_size;
    RecPara rec_para2 = rec_para[(group_idx + 1) * 256 + code2];
    __m128 lower2 = rec_para2.lower;
    __m128 step2 = rec_para2.step_size;

    __m256 lower_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(lower1), lower2, 1);
    __m256 step_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(step1), step2, 1);

    __m256i code_bytes = _mm256_set1_epi32(code_value);
    __m256i shift_amounts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    __m256i shifted_code = _mm256_srlv_epi32(code_bytes, shift_amounts);
    __m256i mask = _mm256_set1_epi32(3);
    __m256i masked_code = _mm256_and_si256(shifted_code, mask);

    __m256 code_vec = _mm256_cvtepi32_ps(masked_code);
    __m256 reconstructed_vec = _mm256_fmadd_ps(code_vec, step_vec, lower_vec);

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
    int group_idx = d / 4;
    CodeUnit code_unit = code[(offset / 4 + group_idx) / 2];
    uint8_t code1 = code_unit.x1;
    uint8_t code2 = code_unit.x2;
    uint16_t code_value = code_unit.code;
    RecPara rec_para1 = rec_para[group_idx * 256 + code1];
    __m128 lower1 = rec_para1.lower;
    __m128 step1 = rec_para1.step_size;
    RecPara rec_para2 = rec_para[(group_idx + 1) * 256 + code2];
    __m128 lower2 = rec_para2.lower;
    __m128 step2 = rec_para2.step_size;

    __m256 lower_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(lower1), lower2, 1);
    __m256 step_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(step1), step2, 1);

    __m256i code_bytes = _mm256_set1_epi32(code_value);
    __m256i shift_amounts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    __m256i shifted_code = _mm256_srlv_epi32(code_bytes, shift_amounts);
    __m256i mask = _mm256_set1_epi32(3);
    __m256i masked_code = _mm256_and_si256(shifted_code, mask);

    __m256 code_vec = _mm256_cvtepi32_ps(masked_code);
    __m256 reconstructed_vec = _mm256_fmadd_ps(code_vec, step_vec, lower_vec);

    _mm256_storeu_ps(dist + d, reconstructed_vec);
  }
}
