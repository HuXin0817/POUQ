#include "../src/quantizer.h"

#include <iostream>
#include <random>
#include <vector>

#include "baseline/scalar_quantizer.h"

std::vector<std::vector<float>> generateRandom2DFloatUnifrom(uint32_t n, uint32_t m, float k) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-k, k);

  std::vector<std::vector<float>> arr(n, std::vector<float>(m));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      arr[i][j] = dist(gen);
    }
  }

  return arr;
}

template <typename Quantizer>
void Benchmark(const char* name, const std::vector<std::vector<float>>& data) {
  Quantizer quantizer;
  quantizer.Train(data);

  float max_decode_diff = 0.0f;
  float sum_decode_diff = 0.0f;
  float sum_decode_power_diff = 0.0f;
  for (size_t i = 0; i < data.size(); ++i) {
    std::vector<float> decode = quantizer.Decode(i);
    for (size_t j = 0; j < data[0].size(); ++j) {
      float diff = std::abs(data[i][j] - decode[j]);
      sum_decode_diff += diff;
      sum_decode_power_diff += diff * diff;
      if (diff > max_decode_diff) {
        max_decode_diff = diff;
      }
    }
  }

  std::cout << name << ":" << std::endl;
  std::cout << "  Max Error: " << max_decode_diff << std::endl;
  std::cout << "  MAE: " << sum_decode_diff / (data.size() * data[0].size()) << std::endl;
  std::cout << "  MSE: " << sum_decode_power_diff / (data.size() * data[0].size()) << std::endl;
}

int main() {
  static constexpr uint32_t kSample = 10000;
  static constexpr uint32_t kDim = 256;
  static constexpr float kRange = 1000.0f;

  std::vector<std::vector<float>> data = generateRandom2DFloatUnifrom(kSample, kDim, kRange);

  Benchmark<ScalarQuantizer>("Scalar Quantizer", data);
  Benchmark<pouq::Quantizer>("POUQ Quantizer", data);
}