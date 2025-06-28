#include "quantizer.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

constexpr size_t N   = 5e4;
constexpr size_t Dim = 256;

float compute_mse(const pouq::Quantizer &quant, const std::vector<float> &data) {
  float err = 0;
  for (size_t i = 0; i < data.size(); i += Dim) {
    err += quant.l2distance(data.data() + i, i);
  }
  return err / static_cast<float>(data.size());
}

int main() {
  std::mt19937                   gen(42);
  std::uniform_real_distribution dis(0.0f, 1.0f);

  std::vector<float> data(N * Dim);
#pragma omp parallel for
  for (auto &d : data) {
    d = dis(gen);
  }

  pouq::Quantizer quantizer(Dim);

  {
    const auto start_time = std::chrono::high_resolution_clock::now();
    quantizer.train(data.data(), data.size());
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << std::left << std::setw(18) << "Training time:" << duration.count() << "s" << std::endl;
  }
  {
    const auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << std::left << std::setw(18) << "Error:" << compute_mse(quantizer, data) << std::endl;
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << std::left << std::setw(18) << "QPS:" << N / duration.count() << " vec/s" << std::endl;
  }
}
