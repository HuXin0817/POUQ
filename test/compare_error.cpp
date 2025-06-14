#include "../libpouq/quantizer.hpp"
#include "sq8impl.hpp"
#include "utils.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

constexpr auto now = std::chrono::high_resolution_clock::now;

template <typename T>
float compute_mse(size_t Dim, const T &quant, const std::vector<float> &data) {
  float err = 0;
  for (size_t i = 0; i < data.size(); i += Dim) {
    err += quant.l2distance(data.data() + i, i);
  }
  return err / static_cast<float>(data.size());
}

template <typename T>
void run(const std::string &method_name, size_t Dim, const std::vector<float> &data) {
  std::cout << std::left << std::setw(16) << "Method Name:" << method_name << std::endl;
  T    quantizer(Dim);
  auto start_time = now();
  quantizer.train(data.data(), data.size());
  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(now() - start_time);
  std::cout << std::left << std::setw(16) << "Training Time:" << duration.count() << " second" << std::endl;
  start_time = now();
  std::cout << std::left << std::setw(16) << "Error:" << compute_mse(Dim, quantizer, data) << std::endl;
  duration = std::chrono::duration_cast<std::chrono::duration<double>>(now() - start_time);
  std::cout << std::left << std::setw(16) << "QPS:" << static_cast<float>(data.size() / Dim) / duration.count()
            << " vector/second" << std::endl
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <dataset_name>" << std::endl;
    exit(0);
  }

  const std::string dataset = argv[1];
  const auto [Dim, data]    = read_fvecs("../data/" + dataset + "/" + dataset + "_base.fvecs");

  run<SQ8Quantizer>("SQ", Dim, data);
  run<pouq::Quantizer>("POUQ", Dim, data);
}
