#pragma once

#include <cmath>
#include <limits>
#include <vector>

class SQQuantizer {
public:
  explicit SQQuantizer(size_t dim) : nbit(4), dim(dim) {}

  void train(const float *data, size_t data_size) {
    encode    = new uint8_t[(data_size * nbit + 7) / 8];
    codebook  = new std::pair<float, float>[dim];
    float div = (1 << nbit) - 1;

#pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
      codebook[i].first  = std::numeric_limits<float>::min();
      codebook[i].second = std::numeric_limits<float>::max();
      for (size_t j = i; j < data_size; j += dim) {
        codebook[i].first  = std::max(codebook[i].first, data[j]);
        codebook[i].second = std::min(codebook[i].second, data[j]);
      }
      if (codebook[i].first == codebook[i].second) {
        codebook[i].second = 1.0f;
      } else {
        codebook[i].second = (codebook[i].second - codebook[i].first) / div;
      }
      for (size_t j = i; j < data_size; j += dim) {
        set(encode, j, std::round((data[j] - codebook[i].first) / codebook[i].second), nbit);
      }
    }
  }

  float l2distance(const float *data, size_t data_index) const {
    float dis = 0.0f;
    for (size_t i = 0; i < dim; i++) {
      float diff =
          static_cast<float>(get(encode, data_index + i, nbit)) * codebook[i].second + codebook[i].first - data[i];
      dis += diff * diff;
    }
    return dis;
  }

private:
  size_t                   nbit;
  size_t                   dim;
  std::pair<float, float> *codebook = nullptr;
  uint8_t                 *encode   = nullptr;

  void set(uint8_t *data, size_t index, size_t n, size_t bit_size) {
    if (bit_size == 0) {
      return;
    }

    n &= (1 << bit_size) - 1;
    const size_t pos = index * bit_size;
    for (size_t bit = 0; bit < bit_size; ++bit) {
      const size_t i      = (pos + bit) / 8;
      const size_t offset = (pos + bit) % 8;
      if (n & 1 << bit) {
        data[i] |= 1 << offset;
      } else {
        data[i] &= ~(1 << offset);
      }
    }
  }

  size_t get(const uint8_t *data, size_t index, size_t bit_size) const {
    if (bit_size == 0) {
      return 0;
    }

    const size_t pos    = index * bit_size;
    size_t       result = 0;
    for (size_t bit = 0; bit < bit_size; ++bit) {
      const size_t i      = (pos + bit) / 8;
      const size_t offset = (pos + bit) % 8;
      if (data[i] & 1 << offset) {
        result |= 1 << bit;
      }
    }

    return result;
  }
};
