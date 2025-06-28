#include "../libpouq/quantizer.hpp"

#include <assert.h>
#include <chrono>
#include <iomanip>
#include <iostream>

constexpr size_t Dim = 256;
constexpr size_t N   = 1e4 * Dim;

float compute_mse(const std::vector<float> &d1, const pouq::Quantizer &d2, size_t size) {
  float mse = 0;
  for (size_t i = 0; i < size; i += Dim) {
    mse += d2.l2distance(d1.data() + i, i);
  }
  return mse / static_cast<float>(size);
}

int main() {
  std::mt19937                   gen(42);
  std::uniform_real_distribution dis(0.0f, 256.0f);

  std::vector<float> data(N);
  for (auto &d : data) {
    d = dis(gen);
  }

  pouq::Quantizer quantizer(Dim);

  const auto start_time = std::chrono::high_resolution_clock::now();
  quantizer.train(data.data(), N);
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

  std::cout << std::left << std::setw(18) << "Training time:" << duration.count() << "s" << std::endl;

  const auto mse_start_time = std::chrono::high_resolution_clock::now();
  float      error          = compute_mse(data, quantizer, N);
  const auto mse_end_time   = std::chrono::high_resolution_clock::now();
  const auto mse_duration   = std::chrono::duration_cast<std::chrono::duration<double>>(mse_end_time - mse_start_time);

  std::cout << std::left << std::setw(18) << "MSE compute time:" << mse_duration.count() << "s" << std::endl;
  std::cout << std::left << std::setw(18) << "Error:" << error << std::endl;

  return 0;
}
