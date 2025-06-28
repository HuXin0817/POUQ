#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename D1, typename D2, typename T>
float compute_mse(const D1 &d1, const D2 &d2, T size) {
  float mse = 0;
  for (T i = 0; i < size; ++i) {
    const float dif = d1[i] - d2[i];
    mse += dif * dif;
  }
  return mse / static_cast<float>(size);
}

std::vector<float> read_fvecs(const std::string &filename) {
  std::cout << "reading from file " << filename << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    exit(-1);
  }
  std::vector<float> all_data;
  int                dim;
  while (file.read(reinterpret_cast<char *>(&dim), sizeof(int))) {
    if (dim <= 0) {
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
  return all_data;
}
