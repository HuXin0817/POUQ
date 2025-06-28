#include "libpouq/quantizer.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

constexpr auto now = std::chrono::high_resolution_clock::now;

float compute_mse(size_t Dim, const pouq::Quantizer &quant, const std::vector<float> &data) {
  float err = 0;
  for (size_t i = 0; i < data.size(); i += Dim) {
    err += quant.l2distance(data.data() + i, i);
  }
  return err / static_cast<float>(data.size());
}

std::pair<size_t, std::vector<float>> read_fvecs(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  std::vector<float> data;
  size_t             dim = 0;

  int first_dim;
  if (file.read(reinterpret_cast<char *>(&first_dim), sizeof(int))) {
    dim = static_cast<size_t>(first_dim);

    std::vector<float> first_vector(dim);
    if (!file.read(reinterpret_cast<char *>(first_vector.data()), dim * sizeof(float))) {
      throw std::runtime_error("Error reading first vector data");
    }

    data.insert(data.end(), first_vector.begin(), first_vector.end());

    int current_dim;
    while (file.read(reinterpret_cast<char *>(&current_dim), sizeof(int))) {
      if (static_cast<size_t>(current_dim) != dim) {
        throw std::runtime_error("Inconsistent vector dimensions in file");
      }

      std::vector<float> vector(dim);
      if (!file.read(reinterpret_cast<char *>(vector.data()), dim * sizeof(float))) {
        throw std::runtime_error("Error reading vector data");
      }

      data.insert(data.end(), vector.begin(), vector.end());
    }
  } else {
    throw std::runtime_error("Empty file or error reading dimension");
  }

  file.close();
  return {dim, data};
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <dataset_name>" << std::endl;
    exit(0);
  }

  const std::string dataset = argv[1];
  const auto [Dim, data]    = read_fvecs("../data/" + dataset + "/" + dataset + "_base.fvecs");

  pouq::Quantizer quantizer(Dim);

  auto start_time = now();
  quantizer.train(data.data(), data.size());
  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(now() - start_time);
  std::cout << std::left << std::setw(18) << "Training time:" << duration.count() << "s" << std::endl;
  start_time = now();
  std::cout << std::left << std::setw(18) << "Error:" << compute_mse(Dim, quantizer, data) << std::endl;
  duration = std::chrono::duration_cast<std::chrono::duration<double>>(now() - start_time);
  std::cout << std::left << std::setw(18) << "QPS:" << static_cast<float>(data.size() / Dim) / duration.count()
            << " vec/s" << std::endl;
}
