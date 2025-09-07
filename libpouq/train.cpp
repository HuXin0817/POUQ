#include "train.h"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

bool
train_impl(int dim,
           CodeUnit* code,
           RecPara* rec_para,
           const float* data,
           int size,
           const Parameter& parameter) {
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
  bool success = true;

  do_malloc(steps, float, dim * 4);
  do_malloc(lowers, float, dim * 4);
  do_malloc(cid, uint8_t, size);
  do_malloc(codes, uint8_t, size);
  do_malloc(segment_lower, float, dim * 4);
  do_malloc(segment_upper, float, dim * 4);
  do_malloc(train_data_map, float, size);
  do_malloc(train_freq_map, int, size);

#pragma omp parallel for
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
        segment(4, data_map, freq_map, data_map_size, do_count_freq, seg_lower, seg_upper);
    if (segment_size == 0) {
      success = false;
      continue;
    }

    for (int i = 0; i < segment_size; i++) {
      float lower = seg_lower[i];
      float upper = seg_upper[i];
      if (lower < upper) {
        float* data_begin = std::lower_bound(data_map, data_map + data_map_size, lower);
        float* data_end = std::upper_bound(data_map, data_map + data_map_size, upper);

        int data_index = data_begin - data_map;

        std::tie(lower, upper) = optimize(3,
                                          lower,
                                          upper,
                                          data_begin,
                                          freq_map + data_index,
                                          data_end - data_begin,
                                          parameter,
                                          do_count_freq);
      }
      lowers[d * 4 + i] = lower;
      if (lower == upper) {
        steps[d * 4 + i] = 1.0;
      } else {
        steps[d * 4 + i] = (upper - lower) / 3.0f;
      }
    }

    for (int i = d; i < size; i += dim) {
      float* it = std::upper_bound(seg_lower, seg_lower + segment_size, data[i]);
      int c = it - seg_lower - 1;
      float x = std::clamp((data[i] - lowers[d * 4 + c]) / steps[d * 4 + c] + 0.5f, 0.0f, 3.0f);
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

    code[i] = std::make_tuple(
        x0 | x1 | x2 | x3, x4 | x5 | x6 | x7, x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15);
  }

#pragma omp parallel for
  for (int g = 0; g < dim / 4; g++) {
    for (int j = 0; j < 256; j++) {
      int x0 = g * 16 + 0 * 4 + (j & 3);
      int x1 = g * 16 + 1 * 4 + (j >> 2 & 3);
      int x2 = g * 16 + 2 * 4 + (j >> 4 & 3);
      int x3 = g * 16 + 3 * 4 + (j >> 6 & 3);
      set_rec_para(&rec_para[g * 256 + j],
                   lowers[x0],
                   lowers[x1],
                   lowers[x2],
                   lowers[x3],
                   steps[x0],
                   steps[x1],
                   steps[x2],
                   steps[x3]);
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

  return true;
}

std::pair<CodeUnit*, RecPara*>
train(int dim, const float* data, int size, const Parameter& parameter) {
  CodeUnit* code_ = nullptr;
  RecPara* rec_para = nullptr;

  do_malloc(code_, CodeUnit, size / 8);
  do_malloc(rec_para, RecPara, dim * 64);
  if (!train_impl(dim, code_, rec_para, data, size, parameter)) {
    goto cleanup;
  }

  return {code_, rec_para};

cleanup:
  do_free(code_);
  do_free(rec_para);

  return {nullptr, nullptr};
}
