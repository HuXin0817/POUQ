#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <immintrin.h>  // 添加SIMD头文件

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
