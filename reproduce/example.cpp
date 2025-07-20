#include "../libpouq/quantizer.hpp"
#include "../libpouq/utils.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

template <typename Quantizer>
float compute_mse(const std::vector<float> &d1, const Quantizer &d2, size_t dim) {
  float mse = 0;
#pragma omp parallel for reduction(+ : mse)
  for (size_t i = 0; i < d1.size(); i += dim) {
    mse += d2.l2distance(d1.data() + i, i);
  }
  return mse / static_cast<float>(d1.size());
}

template <typename Quantizer>
std::vector<std::string> run(size_t dim, const std::vector<float> &data, const std::string &quantizer_name) {
  Quantizer quantizer(dim);

  const auto start_time = std::chrono::high_resolution_clock::now();
  quantizer.train(data.data(), data.size());
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

  float  error;
  double mse_duration = 0.0;
  for (size_t i = 0; i < 10; i++) {
    const auto mse_start_time = std::chrono::high_resolution_clock::now();
    error                     = compute_mse(data, quantizer, dim);
    const auto mse_end_time   = std::chrono::high_resolution_clock::now();
    mse_duration += std::chrono::duration_cast<std::chrono::duration<double>>(mse_end_time - mse_start_time).count();
  }
  mse_duration /= 10;

  double decode_speed = static_cast<double>(data.size()) / dim / mse_duration;

  std::cout << "Quantizer: " << quantizer_name << std::endl;
  std::cout << std::left << std::setw(18) << "Training time:" << duration.count() << "s" << std::endl;
  std::cout << std::left << std::setw(18) << "Decode Speed:" << decode_speed << " vector/s" << std::endl;
  std::cout << std::left << std::setw(18) << "Error:" << error << std::endl;
  std::cout << std::endl;

  std::ostringstream duration_ss, speed_ss, error_ss;
  duration_ss << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << duration.count();
  speed_ss << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << decode_speed;
  error_ss << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) << error;

  return {quantizer_name, duration_ss.str(), speed_ss.str(), error_ss.str()};
}

void write_to_csv(const std::vector<std::vector<std::string>> &rows, const std::string &filename) {
  std::ofstream csv_file(filename);
  if (!csv_file.is_open()) {
    std::cerr << "Error: Cannot create CSV file " << filename << std::endl;
    exit(-1);
  }

  // 设置全精度输出
  csv_file << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);

  // 写入表头
  csv_file << "Quantizer,Training Time (s),Decode Speed (vector/s),Error\n";

  // 写入数据行
  for (const auto &row : rows) {
    for (size_t i = 0; i < row.size(); ++i) {
      csv_file << row[i];
      if (i < row.size() - 1)
        csv_file << ",";
    }
    csv_file << "\n";
  }

  csv_file.close();
  std::cout << "Results saved to " << filename << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <dataset_name> [output_csv]" << std::endl;
    return -1;
  }

  const std::string dataset_name = argv[1];
  const std::string csv_filename = (argc > 2) ? argv[2] : ("../result/exp1_" + dataset_name + ".csv");

  auto  d1   = read_fvecs("../data/" + dataset_name + "/" + dataset_name + "_base.fvecs");
  auto &data = d1.first;
  auto  dim  = d1.second;

  std::vector<std::vector<std::string>> results;

  results.push_back(run<UQ4bitSIMDQuantizer>(dim, data, "UQ4bitSIMD"));
  results.push_back(run<POUQ4bitSIMDQuantizer>(dim, data, "POUQ4bitSIMD"));

  results.push_back(run<UQQuantizer<4, MinMaxOptimizer>>(dim, data, "UQ4bit"));
  results.push_back(run<UQQuantizer<8, MinMaxOptimizer>>(dim, data, "UQ8bit"));

  results.push_back(run<UQQuantizer<4, EMOptimizer>>(dim, data, "UQ4bitEMOptimize"));
  results.push_back(run<UQQuantizer<8, EMOptimizer>>(dim, data, "UQ8bitEMOptimize"));

  results.push_back(run<UQQuantizer<4, PSOOptimizer>>(dim, data, "UQ4bitPSOOptimize"));
  results.push_back(run<UQQuantizer<8, PSOOptimizer>>(dim, data, "UQ8bitPSOOptimize"));

  results.push_back(run<POUQQuantizer<4, POUQSegmenter, MinMaxOptimizer>>(dim, data, "POUQ4bitMinMax"));
  results.push_back(run<POUQQuantizer<8, POUQSegmenter, MinMaxOptimizer>>(dim, data, "POUQ8bitMinMax"));

  results.push_back(run<POUQQuantizer<4, KmeansSegmenter, MinMaxOptimizer>>(dim, data, "POUQ4bitKmeansMinMax"));
  results.push_back(run<POUQQuantizer<8, KmeansSegmenter, MinMaxOptimizer>>(dim, data, "POUQ8bitKmeansMinMax"));

  results.push_back(run<POUQQuantizer<4, POUQSegmenter, PSOOptimizer>>(dim, data, "POUQ4bitPSOOptimize"));
  results.push_back(run<POUQQuantizer<8, POUQSegmenter, PSOOptimizer>>(dim, data, "POUQ8bitPSOOptimize"));

  results.push_back(run<POUQQuantizer<4, POUQSegmenter, EMOptimizer>>(dim, data, "POUQ4bitEMOptimize"));
  results.push_back(run<POUQQuantizer<8, POUQSegmenter, EMOptimizer>>(dim, data, "POUQ8bitEMOptimize"));

  write_to_csv(results, csv_filename);

  return 0;
}