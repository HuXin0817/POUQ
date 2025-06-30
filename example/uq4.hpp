#pragma once

#include <algorithm>
#include <cassert>
#include <immintrin.h>  // AVX2和FMA指令集头文件
#include <omp.h>

class UQQuantizer {

public:
  explicit UQQuantizer(size_t groups) : dim_(groups) {
    assert(dim_ % 32 == 0);
    // 预计算解包掩码
    unpack_mask = _mm256_set1_epi32(0x0F0F0F0F);
    shift_vec   = _mm256_setr_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  }

  ~UQQuantizer() { delete[] code; }

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

    // 初始化累加器
    __m256       acc       = _mm256_setzero_ps();
    const __m256 scale_vec = _mm256_set1_ps(step_size);
    const __m256 lower_vec = _mm256_set1_ps(lower);

    for (size_t i = 0; i < dim_; i += 8) {
      // 加载压缩的4-bit值
      const uint32_t packed     = code[offset + i / 8];
      __m256i        packed_vec = _mm256_set1_epi32(packed);

      // 解包4-bit值到8个32位整数
      __m256i shifted        = _mm256_srlv_epi32(packed_vec, shift_vec);
      __m256i quantized_ints = _mm256_and_si256(shifted, unpack_mask);

      // 转换为浮点数并反量化
      __m256 dequantized = _mm256_cvtepi32_ps(quantized_ints);
      dequantized        = _mm256_fmadd_ps(dequantized, scale_vec, lower_vec);

      // 加载输入向量
      __m256 input_vec = _mm256_loadu_ps(data + i);

      // 计算差值并平方累加
      __m256 diff = _mm256_sub_ps(dequantized, input_vec);
      acc         = _mm256_fmadd_ps(diff, diff, acc);
    }

    // 水平求和累加器
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    __m128 hi64   = _mm_unpackhi_ps(sum128, sum128);
    __m128 lo64   = _mm_add_ps(sum128, hi64);
    __m128 hi32   = _mm_movehl_ps(lo64, lo64);
    __m128 sum    = _mm_add_ss(lo64, hi32);
    ret           = _mm_cvtss_f32(sum);

    return ret;
  }

private:
  size_t dim_ = 0;
  float  lower;
  float  step_size;

  // 用于SIMD操作的掩码和移位向量
  __m256i unpack_mask;
  __m256i shift_vec;

  uint32_t *code = nullptr;
};