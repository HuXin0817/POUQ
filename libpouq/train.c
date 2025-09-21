#include "train.h"

int
train_impl(int dim,
           CodeUnit* code,
           RecPara* rec_para,
           const float* data,
           int size,
           const Parameter parameter) {
  assert(data != nullptr);
  assert(size > 0);
  assert(size % dim == 0);

  float* steps = nullptr;
  float* lowers = nullptr;
  uint8_t* cid = nullptr;
  uint8_t* codes = nullptr;
  float* segment_lower = nullptr;
  float* segment_upper = nullptr;
  float* train_data_map = nullptr;
  int* train_freq_map = nullptr;
  int success = 1;

  do_malloc(steps, float, dim * 4);
  do_malloc(lowers, float, dim * 4);
  do_malloc(cid, uint8_t, size);
  do_malloc(codes, uint8_t, size);
  do_malloc(segment_lower, float, dim * 4);
  do_malloc(segment_upper, float, dim * 4);
  do_malloc(train_data_map, float, size);
  do_malloc(train_freq_map, int, size);

  for (int d = 0; d < dim; d++) {
    float* data_map = train_data_map + d * (size / dim);
    int* freq_map = train_freq_map + d * (size / dim);
    int data_map_size = get_sorted_data(data, size, d, dim, data_map);

    bool do_count_freq = data_map_size < size * 8 / dim / 10;
    if (do_count_freq) {
      data_map_size = count_freq(data_map, data_map_size, data_map, freq_map);
    }

    float* seg_lower = segment_lower + d * 4;
    float* seg_upper = segment_upper + d * 4;

    int segment_size =
        segment(data_map, freq_map, data_map_size, do_count_freq, seg_lower, seg_upper);
    if (segment_size == 0) {
      success = 0;
      continue;
    }

    for (int i = 0; i < segment_size; i++) {
      float lower = seg_lower[i];
      float upper = seg_upper[i];
      if (lower < upper) {
        float* data_begin = data_map;
        for (int i = 0; i < data_map_size; i++) {
          if (data_map[i] >= lower) {
            data_begin = data_map + i;
            break;
          }
        }
        float* data_end = data_map + data_map_size;
        for (int i = 0; i < data_map_size; i++) {
          if (data_map[i] > upper) {
            data_end = data_map + i;
            break;
          }
        }

        int data_index = data_begin - data_map;

        Bound bound = optimize(lower,
                               upper,
                               data_begin,
                               freq_map + data_index,
                               data_end - data_begin,
                               parameter,
                               do_count_freq);
        lower = bound.lower;
        upper = bound.upper;
      }
      lowers[d * 4 + i] = lower;
      if (lower == upper) {
        steps[d * 4 + i] = 1.0;
      } else {
        steps[d * 4 + i] = (upper - lower) / DIV;
      }
    }

    for (int i = d; i < size; i += dim) {
      int c = 0;
      for (int j = 0; j < segment_size; j++) {
        if (data[i] <= seg_lower[j]) {
          c = j - 1;
          break;
        }
        c = j;
      }
      if (c < 0)
        c = 0;
      if (c >= segment_size)
        c = segment_size - 1;

      float x = (data[i] - lowers[d * 4 + c]) / steps[d * 4 + c] + 0.5f;
      if (x < 0.0f) {
        x = 0.0f;
      }
      if (x > DIV) {
        x = DIV;
      }
      cid[i] = c;
      codes[i] = (uint8_t)(x);
    }
  }

  if (!success) {
    goto cleanup;
  }

#pragma omp parallel for
  for (int i = 0; i < size / 8; i++) {
    uint8_t x0 = (cid[i * 8] & 3) << 0;
    uint8_t x1 = (cid[i * 8 + 1] & 3) << 2;
    uint8_t x2 = (cid[i * 8 + 2] & 3) << 4;
    uint8_t x3 = (cid[i * 8 + 3] & 3) << 6;
    uint8_t x4 = (cid[i * 8 + 4] & 3) << 0;
    uint8_t x5 = (cid[i * 8 + 5] & 3) << 2;
    uint8_t x6 = (cid[i * 8 + 6] & 3) << 4;
    uint8_t x7 = (cid[i * 8 + 7] & 3) << 6;

    uint16_t x8 = (codes[i * 8] & 3) << 0;
    uint16_t x9 = (codes[i * 8 + 1] & 3) << 2;
    uint16_t x10 = (codes[i * 8 + 2] & 3) << 4;
    uint16_t x11 = (codes[i * 8 + 3] & 3) << 6;
    uint16_t x12 = (codes[i * 8 + 4] & 3) << 8;
    uint16_t x13 = (codes[i * 8 + 5] & 3) << 10;
    uint16_t x14 = (codes[i * 8 + 6] & 3) << 12;
    uint16_t x15 = (codes[i * 8 + 7] & 3) << 14;

    code[i].x1 = x0 | x1 | x2 | x3;
    code[i].x2 = x4 | x5 | x6 | x7;
    code[i].code = x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15;
  }

#pragma omp parallel for
  for (int g = 0; g < dim / 4; g++) {
    for (int j = 0; j < 256; j++) {
      int x0 = g * 16 + 0 * 4 + (j & 3);
      int x1 = g * 16 + 1 * 4 + (j >> 2 & 3);
      int x2 = g * 16 + 2 * 4 + (j >> 4 & 3);
      int x3 = g * 16 + 3 * 4 + (j >> 6 & 3);
      rec_para[g * 256 + j].lower = _mm_setr_ps(lowers[x0], lowers[x1], lowers[x2], lowers[x3]);
      rec_para[g * 256 + j].step_size = _mm_setr_ps(steps[x0], steps[x1], steps[x2], steps[x3]);
    }
  }

cleanup:
  do_free(train_freq_map);
  do_free(train_data_map);
  do_free(segment_upper);
  do_free(segment_lower);
  do_free(codes);
  do_free(cid);
  do_free(lowers);
  do_free(steps);

  return success;
}

Result
train(int dim, const float* data, int size, const Parameter parameter) {
  CodeUnit* code_ = nullptr;
  RecPara* rec_para = nullptr;

  do_malloc(code_, CodeUnit, size / 8);
  do_malloc(rec_para, RecPara, dim * 64);
  if (!train_impl(dim, code_, rec_para, data, size, parameter)) {
    goto cleanup;
  }

  Result result;
  result.code = code_;
  result.rec_para = rec_para;
  return result;

cleanup:
  do_free(code_);
  do_free(rec_para);

  Result error_result;
  error_result.code = nullptr;
  error_result.rec_para = nullptr;
  return error_result;
}
