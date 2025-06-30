#pragma once

#include <algorithm>
#include <immintrin.h>  // AVX2指令集
#include <omp.h>

#include <cassert>

class UQQuantizer {

public:
  explicit UQQuantizer(size_t groups) : dim_(groups) { assert(dim_ % 32 == 0); }

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

    // AVX2寄存器用于存储常量
    const __m256 lower_vec     = _mm256_set1_ps(lower);
    const __m256 step_size_vec = _mm256_set1_ps(step_size);
    __m256       sum_vec       = _mm256_setzero_ps();

    for (size_t i = 0; i < dim_; i += 8) {
      const uint32_t packed = code[offset + i / 8];

      // 方法1: 使用AVX2指令快速解包
      const __m256i packed_vec    = _mm256_set1_epi32(packed);
      const __m256i shift_amounts = _mm256_setr_epi32(28, 24, 20, 16, 12, 8, 4, 0);
      const __m256i mask          = _mm256_set1_epi32(0xF);

      // 并行右移和掩码操作
      __m256i shifted    = _mm256_srlv_epi32(packed_vec, shift_amounts);
      __m256i q_vals_vec = _mm256_and_si256(shifted, mask);

      // 转换为浮点数
      __m256 q_float = _mm256_cvtepi32_ps(q_vals_vec);

      // 加载输入数据
      __m256 data_vec = _mm256_loadu_ps(&data[i]);

      // 计算: lower + step_size * q - data
      __m256 decoded = _mm256_fmadd_ps(step_size_vec, q_float, lower_vec);
      __m256 diff    = _mm256_sub_ps(decoded, data_vec);

      // 计算平方并累加
      sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    // 水平求和AVX2寄存器中的8个浮点数
    // 水平求和AVX2寄存器中的8个浮点数 - 优化版本
    // 第一步：将256位寄存器分成两个128位部分并相加
    __m128 low128  = _mm256_castps256_ps128(sum_vec);    // 低128位
    __m128 high128 = _mm256_extractf128_ps(sum_vec, 1);  // 高128位
    __m128 sum128  = _mm_add_ps(low128, high128);        // 相加得到4个元素

    // 第二步：水平求和4个元素
    __m128 shuf = _mm_movehdup_ps(sum128);      // 复制奇数位置元素
    sum128      = _mm_add_ps(sum128, shuf);     // 相邻元素相加
    shuf        = _mm_movehl_ps(shuf, sum128);  // 移动高位到低位
    sum128      = _mm_add_ss(sum128, shuf);     // 最终求和

    ret += _mm_cvtss_f32(sum128);  // 提取结果

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