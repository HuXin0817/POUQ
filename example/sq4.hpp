#pragma once

#include <algorithm>
#include <omp.h>

#include <cassert>

class SQ4Quantizer {

public:
  explicit SQ4Quantizer(size_t groups) : dim_(groups) { assert(dim_ % 32 == 0); }

  ~SQ4Quantizer() { delete[] code; }

  void train(const float *data, size_t size) {
    // 计算数据范围
    lower     = data[0];
    step_size = data[0];
    for (size_t i = 1; i < size; i++) {
      lower     = std::min(lower, data[i]);
      step_size = std::max(step_size, data[i]);
    }
    if (lower == step_size) {
      step_size = 1.0;
    } else {
      step_size = (step_size - lower) / 15;
    }

    // 分配内存，每32位存储8个4-bit量化值
    code = new uint32_t[size / 8];

    // 每8个值进行一次量化，打包到一个uint32_t中
    for (size_t i = 0; i < size; i += 8) {
      uint32_t packed = 0;
      for (int j = 0; j < 8 && i + j < size; j++) {
        uint32_t q = std::min<uint32_t>(15, std::max<uint32_t>(0, std::round((data[i + j] - lower) / step_size)));
        packed |= (q << ((7 - j) * 4));
      }
      code[i / 8] = packed;
    }
  }

  float l2distance(const float *data, size_t offset) const {
    float ret = 0.0f;
    offset /= 8;  // 每个uint32_t存储8个值

    for (size_t i = 0; i < dim_; i += 8) {
      const uint32_t packed = code[offset + i / 8];

      // 快速解包8个4-bit值
      const uint32_t q0 = (packed >> 28) & 0xF;
      const uint32_t q1 = (packed >> 24) & 0xF;
      const uint32_t q2 = (packed >> 20) & 0xF;
      const uint32_t q3 = (packed >> 16) & 0xF;
      const uint32_t q4 = (packed >> 12) & 0xF;
      const uint32_t q5 = (packed >> 8) & 0xF;
      const uint32_t q6 = (packed >> 4) & 0xF;
      const uint32_t q7 = packed & 0xF;

      // 解码为浮点数
      const float d0 = lower + step_size * q0 - data[i + 0];
      const float d1 = lower + step_size * q1 - data[i + 1];
      const float d2 = lower + step_size * q2 - data[i + 2];
      const float d3 = lower + step_size * q3 - data[i + 3];
      const float d4 = lower + step_size * q4 - data[i + 4];
      const float d5 = lower + step_size * q5 - data[i + 5];
      const float d6 = lower + step_size * q6 - data[i + 6];
      const float d7 = lower + step_size * q7 - data[i + 7];

      ret += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
    }

    return ret;
  }

  void l2distance_batch(const float *data, size_t size, float *distance) const {
#pragma omp parallel for
    for (size_t i = 0; i < size; i += dim_) {
      distance[i / dim_] = l2distance(data, i);
    }
  }

private:
  size_t dim_ = 0;
  float  lower;
  float  step_size;

  uint32_t *code = nullptr;
};