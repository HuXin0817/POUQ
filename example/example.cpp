#include "../libpouq/quantizer.hpp"
#include "../libpouq/utils.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using SQ           = pouq::QuantizerImpl<pouq::Clusterer, pouq::MinMaxOptimizer>;
using OSQ_Baseline = pouq::QuantizerImpl<pouq::Clusterer, pouq::SGDOptimizer>;
using OSQ          = pouq::QuantizerImpl<pouq::Clusterer, pouq::PSOptimizer>;
using PUQ_KMeans   = pouq::QuantizerImpl<pouq::KmeansClusterer, pouq::MinMaxOptimizer>;
using PUQ_KRange   = pouq::QuantizerImpl<pouq::KrangeClusterer, pouq::MinMaxOptimizer>;
using POUQ         = pouq::QuantizerImpl<pouq::KrangeClusterer, pouq::PSOptimizer>;
using LloydMax     = pouq::QuantizerImpl<pouq::KmeansClusterer, pouq::CenterCalculator>;

std::ofstream csv_file;

template <typename Quantizer>
void run(const std::string &method_name, size_t dim, std::vector<float> data, size_t cbit, size_t qbit) {
  auto start_time = std::chrono::high_resolution_clock::now();

  auto q = Quantizer(cbit, qbit, dim);
  q.train(data.data(), data.size());

  auto end_time   = std::chrono::high_resolution_clock::now();
  auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  auto mse = compute_mse(q, data, data.size());

  // 写入CSV: method,bitwidth,train_time,mse
  size_t total_bitwidth = cbit + qbit;
  csv_file << method_name << "," << total_bitwidth << "," << train_time << "," << mse << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <dataset_file>" << std::endl;
    return 1;
  }

  auto [data, Dim] = read_fvecs(argv[1]);

  // 从文件路径中提取数据集名称
  std::string dataset_path = argv[1];
  std::string dataset_name = std::filesystem::path(dataset_path).stem().string();

  // 创建结果目录
  std::filesystem::create_directories("../result");

  // 打开CSV文件
  std::string csv_filename = "../result/exp1_" + dataset_name + ".csv";
  csv_file.open(csv_filename);

  if (!csv_file.is_open()) {
    std::cerr << "Error: Cannot open file " << csv_filename << std::endl;
    return 1;
  }

  // 写入CSV表头
  csv_file << "method,bitwidth,train_time,mse" << std::endl;

  pouq::POUQQuantizer quantizer(4, 4, 256);

  for (size_t i = 4; i <= 8; i++) {
#define run_sq(M) run<M>(#M, Dim, data, 0, i)

#define run_pouq(M) run<M>(#M, Dim, data, i - (i / 2), i / 2)

    run_sq(SQ);
    run_sq(OSQ_Baseline);
    run_sq(OSQ);

    run_pouq(PUQ_KMeans);
    run_pouq(PUQ_KRange);
    run_pouq(POUQ);

    run<LloydMax>("LloydMax", Dim, data, i, 0);
  }

  csv_file.close();
  std::cout << "Results saved to: " << csv_filename << std::endl;

  return 0;
}
