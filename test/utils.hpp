#pragma once

#include <fstream>
#include <vector>

std::pair<size_t, std::vector<float>> read_fvecs(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  std::vector<float> data;
  size_t             dim = 0;

  int first_dim;
  if (file.read(reinterpret_cast<char *>(&first_dim), sizeof(int))) {
    dim = static_cast<size_t>(first_dim);

    std::vector<float> first_vector(dim);
    if (!file.read(reinterpret_cast<char *>(first_vector.data()), dim * sizeof(float))) {
      throw std::runtime_error("Error reading first vector data");
    }

    data.insert(data.end(), first_vector.begin(), first_vector.end());

    int current_dim;
    while (file.read(reinterpret_cast<char *>(&current_dim), sizeof(int))) {
      if (static_cast<size_t>(current_dim) != dim) {
        throw std::runtime_error("Inconsistent vector dimensions in file");
      }

      std::vector<float> vector(dim);
      if (!file.read(reinterpret_cast<char *>(vector.data()), dim * sizeof(float))) {
        throw std::runtime_error("Error reading vector data");
      }

      data.insert(data.end(), vector.begin(), vector.end());
    }
  } else {
    throw std::runtime_error("Empty file or error reading dimension");
  }

  file.close();
  return {dim, data};
}
