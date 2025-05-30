#include "../libpouq/quantizer.h"

#include <iomanip>
#include <iostream>

constexpr uint64_t N = 1e6;

float compute_mse(const float *data, const uint64_t size, const pouq::Quantizer &quantizer) {
  float mse = 0;
  for (uint64_t i = 0; i < size; ++i) {
    const float dif = data[i] - quantizer[i];
    mse += dif * dif;
  }
  return mse / static_cast<float>(size);
}

template <typename DataType>
void print_vector(const char *prefix, const DataType &data) {
  std::cout << prefix << "[";
  std::cout << std::fixed << std::setprecision(3);
  for (uint64_t i = 0; i < 5; ++i) {
    std::cout << data[i];
    if (i < 4) {
      std::cout << ", ";
    } else {
      std::cout << "...]\n";
    }
  }
  std::cout << std::defaultfloat;
}

int main() {
  std::random_device             rd;
  std::mt19937                   gen(rd());
  std::uniform_real_distribution dis(0.0f, 256.0f);

  auto *data = new float[N];
  for (uint64_t i = 0; i < N; ++i) {
    data[i] = dis(gen);
  }

  const pouq::POUQuantizer quantizer(data, N, 4, 4, 256, true);
  std::cout << "Error: " << compute_mse(data, N, quantizer) << std::endl;

  print_vector("Origin Vector:    ", data);
  print_vector("Quantized Vector: ", quantizer);

  delete[] data;
}
