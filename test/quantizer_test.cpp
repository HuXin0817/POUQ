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

std::vector<std::vector<float>> generateRandom2DFloatGMM(
    uint32_t n, uint32_t m, float k, uint32_t n_clusters, float sigma) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> center_dist(-k, k);
  std::uniform_int_distribution<uint32_t> cluster_choice(0, n_clusters - 1);
  std::normal_distribution<float> noise(0.0f, sigma);

  std::vector<std::vector<float>> centers(m, std::vector<float>(n_clusters));
  for (uint32_t j = 0; j < m; ++j) {
    for (uint32_t c = 0; c < n_clusters; ++c) {
      centers[j][c] = center_dist(gen);
    }
  }

  std::vector<std::vector<float>> arr(n, std::vector<float>(m));
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t j = 0; j < m; ++j) {
      uint32_t c = cluster_choice(gen);
      arr[i][j] = centers[j][c] + noise(gen);
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

  std::cout << "=== Uniform U(-" << kRange << ", " << kRange << ") ===" << std::endl;
  {
    std::vector<std::vector<float>> data = generateRandom2DFloatUnifrom(kSample, kDim, kRange);
    Benchmark<ScalarQuantizer>("Scalar Quantizer", data);
    Benchmark<pouq::Quantizer>("POUQ Quantizer", data);
  }

  struct GMMCase {
    uint32_t clusters;
    float sigma;
  };
  GMMCase cases[] = {
      {2, 20.0f},
      {4, 100.0f},
      {4, 20.0f},
      {4, 5.0f},
      {8, 20.0f},
  };

  for (const auto& c : cases) {
    std::cout << std::endl;
    std::cout << "=== GMM (" << c.clusters << " clusters/dim, sigma=" << c.sigma << ") ===" << std::endl;
    std::vector<std::vector<float>> data = generateRandom2DFloatGMM(kSample, kDim, kRange, c.clusters, c.sigma);
    Benchmark<ScalarQuantizer>("Scalar Quantizer", data);
    Benchmark<pouq::Quantizer>("POUQ Quantizer", data);
  }
}