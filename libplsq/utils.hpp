#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename D1, typename D2, typename T>
float l2distance(const D1 &d1, const D2 &d2, T size) {
  float sum = 0;
  for (T i = 0; i < size; ++i) {
    const float dif = d1[i] - d2[i];
    sum += dif * dif;
  }
  return sum;  // 移除除法，保持与search函数中的计算一致
}

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
