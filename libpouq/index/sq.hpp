#pragma once

#include <cmath>
#include <limits>
#include <vector>

class SQQuantizer {
public:
  explicit SQQuantizer(size_t nbit, size_t dim) : nbit(nbit), dim(dim) {}

  void train(const float *data, size_t data_size) {
    encode    = new uint8_t[data_size];
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
        // pouq::bitmap::set(encode, j, , nbit);
        encode[j] = std::round((data[j] - codebook[i].first) / codebook[i].second);
      }
    }
  }

  float l2distance(const float *data, size_t data_index) const {
    float dis = 0.0f;
    for (size_t i = 0; i < dim; i++) {
      float diff = static_cast<float>(encode[data_index + i]) * codebook[i].second + codebook[i].first - data[i];
      dis += diff * diff;
    }
    return dis;
  }

  ~SQQuantizer() {
    delete[] encode;
    delete[] codebook;
  }

private:
  size_t                   nbit;
  size_t                   dim;
  std::pair<float, float> *codebook = nullptr;
  uint8_t                 *encode   = nullptr;
};
