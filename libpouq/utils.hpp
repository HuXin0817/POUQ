#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <immintrin.h>

std::pair<std::vector<float>, size_t> read_fvecs(const std::string &filename) {
  std::cout << "read from " << filename << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    exit(-1);
  }

  std::vector<float> all_data;
  int                dim        = 0;
  int                first_dim  = -1;
  int                padded_dim = 0;

  while (file.read(reinterpret_cast<char *>(&dim), sizeof(int))) {
    if (dim <= 0) {
      std::cerr << "Error: Invalid dimension " << dim << std::endl;
      exit(-1);
    }

    padded_dim = (dim + 7) / 8 * 8;
    if (first_dim == -1) {
      first_dim = padded_dim;
    } else if (padded_dim != first_dim) {
      std::cerr << "Error: Inconsistent dimensions in fvecs file" << std::endl;
      exit(-1);
    }

    std::vector<float> vec(dim);
    if (!file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float))) {
      std::cerr << "Error: Incomplete vector data" << std::endl;
      exit(-1);
    }

    all_data.insert(all_data.end(), vec.begin(), vec.end());
    if (padded_dim > dim) {
      size_t padding = padded_dim - dim;
      all_data.insert(all_data.end(), padding, 0.0f);
    }
  }

  if (!file.eof()) {
    std::cerr << "Error: Unexpected end of file" << std::endl;
    exit(-1);
  }

  return std::make_pair(all_data, padded_dim);
}

template <typename D1, typename D2, typename T>
float l2distance(const D1 &d1, const D2 &d2, T size) {
  float sum = 0;
  T     i   = 0;

  if constexpr (std::is_same_v<typename std::remove_reference_t<decltype(d1[0])>, float> &&
                std::is_same_v<typename std::remove_reference_t<decltype(d2[0])>, float>) {

    const T simd_size = 8;
    const T simd_end  = (size / simd_size) * simd_size;

    __m256 sum_vec = _mm256_setzero_ps();

    for (; i < simd_end; i += simd_size) {
      __m256 v1   = _mm256_loadu_ps(&d1[i]);
      __m256 v2   = _mm256_loadu_ps(&d2[i]);
      __m256 diff = _mm256_sub_ps(v1, v2);
      sum_vec     = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low  = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128  = _mm_add_ps(sum_low, sum_high);

    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum     = _mm_cvtss_f32(sum_128);
  }

  for (; i < size; ++i) {
    const float dif = d1[i] - d2[i];
    sum += dif * dif;
  }

  return sum;
}
