#include "train.h"

#include <stdlib.h>

void
train_impl(int dim,
           CodeUnit* code,
           RecPara* rec_para,
           const float* data,
           int size,
           const Parameter parameter) {
  assert(data != NULL);
  assert(size > 0);
  assert(size % dim == 0);

  float* steps = malloc(dim * 4 * sizeof(float));
  float* lowers = malloc(dim * 4 * sizeof(float));
  uint8_t* cid = malloc(size * sizeof(uint8_t));
  uint8_t* codes = malloc(size * sizeof(uint8_t));

#pragma omp parallel for
  for (int d = 0; d < dim; d++) {
    float* data_map = malloc(size / dim * sizeof(float));
    int* freq_map = NULL;
    int data_map_size = get_sorted_data(data, size, d, dim, data_map);

    bool do_count_freq = data_map_size < size / dim * 8 / 10;
    if (do_count_freq) {
      freq_map = malloc(size / dim * sizeof(int));
      data_map_size = count_freq(data_map, data_map_size, data_map, freq_map);
    }

    float seg_lower[4];
    float seg_upper[4];

    int seg_size = segment(data_map, freq_map, data_map_size, do_count_freq, seg_lower, seg_upper);

    for (int i = 0; i < seg_size; i++) {
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
        steps[d * 4 + i] = (upper - lower) / 3.0f;
      }
    }

    free(data_map);
    if (do_count_freq) {
      free(freq_map);
    }

    for (int i = d; i < size; i += dim) {
      int c = 0;
      for (int j = 0; j < seg_size; j++) {
        if (data[i] <= seg_lower[j]) {
          c = j - 1;
          break;
        }
        c = j;
      }
      if (c < 0) {
        c = 0;
      }
      if (c >= seg_size) {
        c = seg_size - 1;
      }

      float x = (data[i] - lowers[d * 4 + c]) / steps[d * 4 + c] + 0.5f;
      if (x < 0.0f) {
        x = 0.0f;
      }
      if (x > 3.0f) {
        x = 3.0f;
      }
      cid[i] = c;
      codes[i] = (uint8_t)(x);
    }
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

  free(codes);
  free(cid);
  free(lowers);
  free(steps);
}

Result
train(int dim, const float* data, int size, const Parameter parameter) {
  CodeUnit* code_ = malloc(size / 8 * sizeof(CodeUnit));
  RecPara* rec_para = malloc(dim * 64 * sizeof(RecPara));

  train_impl(dim, code_, rec_para, data, size, parameter);

  Result result;
  result.code = code_;
  result.rec_para = rec_para;
  return result;
}
