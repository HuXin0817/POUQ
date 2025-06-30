#include "../libpouq/quantizer.hpp"
#include "../libpouq/quantizer2.hpp"
#include "uq4.hpp"

#include <assert.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

template <typename Quantizer>
float compute_mse(const std::vector<float> &d1, const Quantizer &d2, size_t size, size_t dim)
{
  float mse = 0;
#pragma omp parallel for reduction(+ : mse)
  for (size_t i = 0; i < size; i += dim)
  {
    mse += d2.l2distance(d1.data() + i, i);
  }
  return mse / static_cast<float>(size);
}

std::pair<std::vector<float>, size_t> read_fvecs(const std::string &filename)
{
  std::cout << "read from " << filename << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    exit(-1);
  }

  std::vector<float> all_data;
  int dim = 0;
  int first_dim = -1;
  int padded_dim = 0;

  while (file.read(reinterpret_cast<char *>(&dim), sizeof(int)))
  {
    if (dim <= 0)
    {
      std::cerr << "Error: Invalid dimension " << dim << std::endl;
      exit(-1);
    }

    // 计算调整后的维度（向上取整到64的倍数）
    padded_dim = ((dim + 7) / 8) * 8;

    // 记录第一个向量的调整后维度
    if (first_dim == -1)
    {
      first_dim = padded_dim;
    }
    else if (padded_dim != first_dim)
    {
      // 如果调整后的维度不一致，报错
      std::cerr << "Error: Inconsistent dimensions in fvecs file" << std::endl;
      exit(-1);
    }

    // 读取原始向量数据
    std::vector<float> vec(dim);
    if (!file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float)))
    {
      std::cerr << "Error: Incomplete vector data" << std::endl;
      exit(-1);
    }

    // 将向量数据添加到结果中
    all_data.insert(all_data.end(), vec.begin(), vec.end());

    // 如果需要，补0到调整后的维度
    if (padded_dim > dim)
    {
      size_t padding = padded_dim - dim;
      all_data.insert(all_data.end(), padding, 0.0f);
    }
  }

  if (!file.eof())
  {
    std::cerr << "Error: Unexpected end of file" << std::endl;
    exit(-1);
  }

  return std::make_pair(all_data, padded_dim);
}

template <typename Quantizer>
std::vector<std::string> run(size_t Dim, const std::vector<float> &data, const std::string &quantizerName)
{
  Quantizer quantizer(Dim);

  const auto start_time = std::chrono::high_resolution_clock::now();
  quantizer.train(data.data(), data.size());
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

  const auto mse_start_time = std::chrono::high_resolution_clock::now();
  float error = compute_mse(data, quantizer, data.size(), Dim);
  const auto mse_end_time = std::chrono::high_resolution_clock::now();
  const auto mse_duration = std::chrono::duration_cast<std::chrono::duration<double>>(mse_end_time - mse_start_time);

  double decode_speed = static_cast<double>(data.size()) / Dim / mse_duration.count();

  std::cout << "Quantizer: " << quantizerName << std::endl;
  std::cout << std::left << std::setw(18) << "Training time:" << duration.count() << "s" << std::endl;
  std::cout << std::left << std::setw(18) << "Decode Speed:" << decode_speed << " vector/s" << std::endl;
  std::cout << std::left << std::setw(18) << "Error:" << error << std::endl;
  std::cout << std::endl;

  std::ostringstream duration_ss, speed_ss, error_ss;
  duration_ss << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << duration.count();
  speed_ss << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << decode_speed;
  error_ss << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) << error;

  return {quantizerName, duration_ss.str(), speed_ss.str(), error_ss.str()};
}

void write_to_csv(const std::vector<std::vector<std::string>> &rows, const std::string &filename)
{
  std::ofstream csv_file(filename);
  if (!csv_file.is_open())
  {
    std::cerr << "Error: Cannot create CSV file " << filename << std::endl;
    exit(-1);
  }

  // 设置全精度输出
  csv_file << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);

  // 写入表头
  csv_file << "Quantizer,Training Time (s),Decode Speed (vector/s),Error\n";

  // 写入数据行
  for (const auto &row : rows)
  {
    for (size_t i = 0; i < row.size(); ++i)
    {
      csv_file << row[i];
      if (i < row.size() - 1)
        csv_file << ",";
    }
    csv_file << "\n";
  }

  csv_file.close();
  std::cout << "Results saved to " << filename << std::endl;
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <dataset_name> [output_csv]" << std::endl;
    return -1;
  }

  const std::string dataset_name = argv[1];
  const std::string csv_filename = (argc > 2) ? argv[2] : (dataset_name + "_results.csv");

  auto d1 = read_fvecs("../data/" + dataset_name + "/" + dataset_name + "_base.fvecs");
  auto &data = d1.first;
  auto Dim = d1.second;

  std::vector<std::vector<std::string>> results;
  run<UQQuantizer>(Dim, data, "UQ");
  results.push_back(run<UQQuantizer>(Dim, data, "UQ"));
  results.push_back(run<pouq::Quantizer>(Dim, data, "POUQ"));
  results.push_back(run<POUQ4>(Dim, data, "POUQ_4"));

  write_to_csv(results, csv_filename);

  return 0;
}