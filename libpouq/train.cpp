#include "train.h"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

void
train_impl(int dim_,
           CodeUnit* code_,
           RecPara* rec_para_,
           const float* data,
           int size,
           const Parameter& parameter) {
  assert(data != nullptr);
  assert(size > 0);
  assert(size % dim_ == 0);

  std::vector<float> steps(dim_ * 4);
  std::vector<float> lowers(dim_ * 4);
  std::vector<uint8_t> cid(size);
  std::vector<uint8_t> code(size);

#pragma omp parallel for
  for (int d = 0; d < dim_; d++) {
    std::vector<float> data_map = get_sorted_data(data, size, d, dim_);
    std::vector<int> freq_map;
    bool do_count_freq = data_map.size() < size * 8 / dim_ / 10;

    if (do_count_freq) {
      std::tie(data_map, freq_map) = count_freq(data_map);
    }

    auto bounds = segment(4, data_map.data(), freq_map.data(), data_map.size(), do_count_freq);

    for (int i = 0; i < bounds.size(); i++) {
      auto [lower, upper] = bounds[i];
      if (lower < upper) {
        auto data_begin = std::lower_bound(data_map.data(),
                                           data_map.data() + data_map.size(),
                                           lower,
                                           [](float lhs, float rhs) -> bool { return lhs < rhs; });
        auto data_end = std::upper_bound(data_map.data(),
                                         data_map.data() + data_map.size(),
                                         upper,
                                         [](float lhs, float rhs) -> bool { return lhs < rhs; });

        auto data_index = data_begin - data_map.data();

        std::tie(lower, upper) = optimize(3,
                                          lower,
                                          upper,
                                          data_begin,
                                          freq_map.data() + data_index,
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

    for (int i = d; i < size; i += dim_) {
      auto it = std::upper_bound(
          bounds.begin(),
          bounds.end(),
          data[i],
          [](float lhs, const std::pair<float, float>& rhs) -> bool { return lhs < rhs.first; });
      int c = static_cast<int>(it - bounds.begin()) - 1;
      float x = std::clamp((data[i] - lowers[d * 4 + c]) / steps[d * 4 + c] + 0.5f, 0.0f, 3.0f);
      cid[i] = c;
      code[i] = static_cast<uint8_t>(x);
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

    uint16_t x8 = (code[i * 8] & 3) << 0;
    uint16_t x9 = (code[i * 8 + 1] & 3) << 2;
    uint16_t x10 = (code[i * 8 + 2] & 3) << 4;
    uint16_t x11 = (code[i * 8 + 3] & 3) << 6;
    uint16_t x12 = (code[i * 8 + 4] & 3) << 8;
    uint16_t x13 = (code[i * 8 + 5] & 3) << 10;
    uint16_t x14 = (code[i * 8 + 6] & 3) << 12;
    uint16_t x15 = (code[i * 8 + 7] & 3) << 14;

    code_[i] = std::make_tuple(
        x0 | x1 | x2 | x3, x4 | x5 | x6 | x7, x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15);
  }

#pragma omp parallel for
  for (int g = 0; g < dim_ / 4; g++) {
    for (int j = 0; j < 256; j++) {
      int x0 = g * 16 + 0 * 4 + (j & 3);
      int x1 = g * 16 + 1 * 4 + (j >> 2 & 3);
      int x2 = g * 16 + 2 * 4 + (j >> 4 & 3);
      int x3 = g * 16 + 3 * 4 + (j >> 6 & 3);
      set_rec_para(&rec_para_[g * 256 + j],
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
}

std::pair<CodeUnit*, RecPara*>
train(int dim_, const float* data, int size, const Parameter& parameter) {
  auto code_ = static_cast<CodeUnit*>(aligned_alloc(256, size / 8 * sizeof(CodeUnit)));
  if (!code_) {
    return {nullptr, nullptr};
  }

  auto rec_para_ = static_cast<RecPara*>(aligned_alloc(256, dim_ * 64 * sizeof(RecPara)));
  if (!rec_para_) {
    free(code_);
    return {nullptr, nullptr};
  }

  train_impl(dim_, code_, rec_para_, data, size, parameter);
  return {code_, rec_para_};
}
