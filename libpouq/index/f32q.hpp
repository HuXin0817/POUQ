#pragma once

#include <cmath>
#include <limits>
#include <vector>

class Float32Quantizer {
public:
  explicit Float32Quantizer(size_t dim) : dim(dim) {}

  void train(const float *data, size_t data_size) {
    encode = new float[data_size];
    memcpy(encode, data, data_size * sizeof(float));
  }

  float l2distance(const float *data, size_t data_index) const {
    float dis = 0.0f;
    for (size_t i = 0; i < dim; i++) {
      float diff = data[i] - encode[i + data_index];
      dis += diff * diff;
    }
    return dis;
  }

private:
  size_t nbit;
  size_t dim;
  float *encode = nullptr;
};
