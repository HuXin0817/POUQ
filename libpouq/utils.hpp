#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__i386__) || defined(__x86_64__)

#include <immintrin.h>  // 添加SIMD头文件

// SIMD优化的L2距离计算函数
template <typename D1, typename D2, typename T>
float l2distance(const D1 &d1, const D2 &d2, T size) {
  float sum = 0;
  T     i   = 0;

  // 检查是否支持AVX2并且数据是float类型
  if constexpr (std::is_same_v<typename std::remove_reference_t<decltype(d1[0])>, float> &&
                std::is_same_v<typename std::remove_reference_t<decltype(d2[0])>, float>) {

    // AVX2处理：每次处理8个float
    const T simd_size = 8;
    const T simd_end  = (size / simd_size) * simd_size;

    __m256 sum_vec = _mm256_setzero_ps();

    for (; i < simd_end; i += simd_size) {
      // 加载8个float值
      __m256 v1 = _mm256_loadu_ps(&d1[i]);
      __m256 v2 = _mm256_loadu_ps(&d2[i]);

      // 计算差值
      __m256 diff = _mm256_sub_ps(v1, v2);

      // 计算平方并累加
      sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    // 水平求和AVX2寄存器中的8个值
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low  = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128  = _mm_add_ps(sum_low, sum_high);

    // 继续水平求和
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);

    sum = _mm_cvtss_f32(sum_128);
  }

  // 处理剩余的元素（标量计算）
  for (; i < size; ++i) {
    const float dif = d1[i] - d2[i];
    sum += dif * dif;
  }

  return sum;  // 移除除法，保持与search函数中的计算一致
}

#else

// SIMD优化的L2距离计算函数
template <typename D1, typename D2, typename T>
float l2distance(const D1 &d1, const D2 &d2, T size) {
  float sum = 0;
  for (T i = 0; i < size; ++i) {
    const float dif = d1[i] - d2[i];
    sum += dif * dif;
  }
  return sum;  // 移除除法，保持与search函数中的计算一致
}

#endif

std::pair<std::vector<float>, size_t> read_fvecs(const std::string &filename) {
  std::cout << "read from " << filename << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    exit(-1);
  }
  std::vector<float> all_data;
  int                dim       = 0;
  int                first_dim = -1;

  while (file.read(reinterpret_cast<char *>(&dim), sizeof(int))) {
    if (dim <= 0) {
      exit(-1);
    }

    // 记录第一个向量的维度
    if (first_dim == -1) {
      first_dim = dim;
    } else if (dim != first_dim) {
      // 如果维度不一致，报错
      std::cerr << "Error: Inconsistent dimensions in fvecs file" << std::endl;
      exit(-1);
    }

    std::vector<float> vec(dim);
    if (!file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float))) {
      exit(-1);
    }
    all_data.insert(all_data.end(), vec.begin(), vec.end());
  }

  if (!file.eof()) {
    exit(-1);
  }

  return std::make_pair(all_data, first_dim);
}
