#pragma once

class Quantizer {
public:
  virtual void train(const float *data, size_t data_size) = 0;

  virtual float l2distance(const float *data, size_t data_index) const = 0;

  virtual ~Quantizer() = 0;
};
