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
  assert(dim % 8 == 0);
  assert(size > 0);
  assert(size % dim == 0);

  float* steps = malloc(dim * 4 * sizeof(float));
  float* lowers = malloc(dim * 4 * sizeof(float));

#pragma omp parallel for
  for (int d = 0; d < dim; d++) {
    float* data_map = malloc(size / dim * sizeof(float));
    int* freq_map = NULL;
    int data_map_size = get_sorted_data(data, size, d, dim, data_map);

    bool do_count_freq = count_unique(data_map, data_map_size) < (size / dim * 7 / 10);
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
        {
          int low = 0;
          int high = (int)data_map_size - 1;
          int found_index = -1;
          while (low <= high) {
            int mid = low + ((high - low) >> 1);
            if (data_map[mid] >= lower) {
              found_index = mid;
              high = mid - 1;
            } else {
              low = mid + 1;
            }
          }
          if (found_index != -1) {
            data_begin = data_map + found_index;
          }
        }
        float* data_end = data_map + data_map_size;
        {
          int low = 0;
          int high = data_map_size;
          while (low < high) {
            int mid = low + (high - low) / 2;
            if (data_map[mid] > upper) {
              high = mid;
            } else {
              low = mid + 1;
            }
          }
          if (low < data_map_size) {
            data_end = data_map + low;
          }
        }

        Bound bound = optimize(lower,
                               upper,
                               data_begin,
                               freq_map + (data_begin - data_map),
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
      int low = 0;
      int high = seg_size - 1;
      int first_ge_idx = seg_size;
      while (low <= high) {
        int mid = low + (high - low) / 2;
        if (seg_lower[mid] >= data[i]) {
          first_ge_idx = mid;
          high = mid - 1;
        } else {
          low = mid + 1;
        }
      }
      if (first_ge_idx < seg_size) {
        c = first_ge_idx - 1;
      } else {
        c = seg_size - 1;
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
#pragma omp critical
      {
        if (i / 4 % 2 == 0) {
          code[i / 8].x1 |= c << (2 * (i % 4));
        } else {
          code[i / 8].x2 |= c << (2 * (i % 4));
        }
        code[i / 8].code |= (uint16_t)(x) << (2 * (i % 8));
      }
    }
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
