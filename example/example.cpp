#include "../libpouq/quantizer.h"
#include "../libpouq/utils.h"

#include <chrono>
#include <iomanip>
#include <iostream>

constexpr size_t N = 1e6;

template <typename DataType>
void print_vector(const char *prefix, const DataType &data) {
  std::cout << prefix << "[";
  std::cout << std::fixed << std::setprecision(3);
  for (size_t i = 0; i < 5; ++i) {
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

  std::vector<float> data(N);
  for (auto &d : data) {
    d = dis(gen);
  }

  pouq::POUQuantizer quantizer(4, 4, 256);

  const auto start_time = std::chrono::high_resolution_clock::now();
  quantizer.train(data.data(), N);
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

  std::cout << "Training time: " << std::fixed << std::setprecision(3) << duration.count() << "s" << std::endl;
  std::cout << "Error: " << compute_mse(data, quantizer, N) << std::endl;

  print_vector("Origin Vector:    ", data);
  print_vector("Quantized Vector: ", quantizer);
}
