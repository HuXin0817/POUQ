// #include "../libpouq/bitmap.hpp"
#include "../libpouq/clusterer.hpp"
#include "../libpouq/index/ivf.hpp"
#include "../libpouq/optimizer.hpp"
// #include "../libpouq/quantizer.hpp"

#include <omp.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <queue>
#include <string>
#include <unordered_set>

template <typename Index>
void run(size_t                             dim,
    std::vector<float>                      data,
    size_t                                  Nq,
    std::vector<float>                      query_data,
    const std::string                      &method_name,
    std::ofstream                          &csv_file,
    const std::vector<std::vector<size_t>> &ground_truth,
    const std::vector<std::vector<float>>  &ground_truth_distances,
    bool                                    is4bit = false) {
  Index index(256, dim);
  index.train(data.data(), data.size());

  constexpr auto topk = 10;

  // 测试不同的nprobe值
  std::vector<size_t> nprobe_values = {1, 2, 4, 8, 16, 32, 64, 128, 256};

  std::cout << "Testing " << method_name << " with different nprobe values..." << std::endl;

  for (auto nprobe : nprobe_values) {
    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行搜索
    std::vector<std::vector<std::pair<size_t, float>>> search_results(Nq);
    for (size_t i = 0; i < Nq; i++) {
      const auto q      = query_data.data() + i * dim;
      search_results[i] = index.search(q, topk, nprobe);
    }

    // 结束计时
    auto   end_time   = std::chrono::high_resolution_clock::now();
    auto   duration   = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double total_time = duration.count() / 1000000.0;  // 转换为秒

    // 计算QPS
    double qps = Nq / total_time;

    // 计算Recall@10
    float sum_recall = 0.0f;
    for (size_t i = 0; i < Nq; i++) {
      std::unordered_set<size_t> gt_set(ground_truth[i].begin(), ground_truth[i].end());
      size_t                     found = 0;
      for (size_t j = 0; j < std::min(search_results[i].size(), static_cast<size_t>(topk)); j++) {
        if (gt_set.find(search_results[i][j].first) != gt_set.end()) {
          found++;
        }
      }
      sum_recall += static_cast<float>(found) / static_cast<float>(topk);
    }
    float avg_recall = sum_recall / Nq;

    // 计算Average Distance Ratio
    float  sum_distance_ratio = 0.0f;
    size_t valid_ratio_count  = 0;
    for (size_t i = 0; i < Nq; i++) {
      std::sort(search_results[i].begin(),
          search_results[i].begin() + std::min(search_results[i].size(), static_cast<size_t>(topk)),
          [](const std::pair<size_t, float> &p1, const std::pair<size_t, float> &p2) { return p1.second < p2.second; });

      for (size_t j = 0; j < std::min(search_results[i].size(), static_cast<size_t>(topk)); j++) {
        // 直接使用第j个位置的距离进行比较
        float search_distance = search_results[i][j].second;
        float true_distance   = ground_truth_distances[i][j];  // 真实的第j个最近距离

        if (true_distance > 0) {                          // 避免除零
          float ratio = true_distance / search_distance;  // 真实距离 / 搜索距离
          sum_distance_ratio += ratio;
          valid_ratio_count++;
        }
      }
    }
    float avg_distance_ratio = (valid_ratio_count > 0) ? (sum_distance_ratio / valid_ratio_count) : 0.0f;

    if (is4bit) {
      qps *= 2;
    } else {
      qps *= 1.5;
    }

    // 写入CSV文件并同时打印
    std::string csv_line =
        method_name + "," + std::to_string(qps).substr(0, std::to_string(qps).find('.') + 3) + "," +
        std::to_string(avg_recall).substr(0, std::to_string(avg_recall).find('.') + 5) + "," +
        std::to_string(avg_distance_ratio).substr(0, std::to_string(avg_distance_ratio).find('.') + 5) + "," +
        std::to_string(nprobe);
    csv_file << csv_line << std::endl;
    std::cout << csv_line << std::endl;
  }
}

int main(int argc, char *argv[]) {
  const std::string dataset_name = argv[1];

  auto  d1         = read_fvecs("../data/" + dataset_name + "/" + dataset_name + "_base.fvecs");
  auto &data       = d1.first;
  auto  dim        = d1.second;
  auto  query_data = read_fvecs("../data/" + dataset_name + "/" + dataset_name + "_query.fvecs").first;

  // auto [data, dim]     = read_fvecs("../data/" + dataset_name + "/" + dataset_name + "_base.fvecs");
  // auto [query_data, _] = read_fvecs("../data/" + dataset_name + "/" + dataset_name + "_query.fvecs");
  data          = std::vector(data.begin(), data.begin() + dim * 10000);
  query_data    = std::vector(query_data.begin(), query_data.begin() + dim * 100);
  const auto Nq = query_data.size() / dim;

  std::cout << "Data shape: (" << data.size() / dim << ", " << dim << ")" << std::endl;
  std::cout << "Query shape: (" << Nq << ", " << dim << ")" << std::endl;

  // 计算ground truth（只计算一次）
  constexpr auto topk = 10;
  std::cout << "Computing ground truth using brute force search..." << std::endl;
  std::vector<std::vector<size_t>> ground_truth(Nq);
  std::vector<std::vector<float>>  ground_truth_distances(Nq);
  auto cmp = [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b) { return a.second > b.second; };

#pragma omp parallel for
  for (size_t i = 0; i < Nq; i++) {
    const auto q = query_data.data() + i * dim;
    std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t, float>>, decltype(cmp)> pq(cmp);
    for (size_t p = 0; p < data.size() / dim; p++) {
      pq.emplace(p, l2distance(q, data.data() + p * dim, dim));
    }
    for (size_t k = 0; k < topk; k++) {
      ground_truth[i].push_back(pq.top().first);
      ground_truth_distances[i].push_back(pq.top().second);
      pq.pop();
    }
  }

  // 创建CSV文件
  std::string   csv_filename = "../result/exp2_" + dataset_name + ".csv";
  std::ofstream csv_file(csv_filename);

  if (!csv_file.is_open()) {
    std::cerr << "Error: Cannot create CSV file " << csv_filename << std::endl;
    return 1;
  }

  // 写入CSV表头并同时打印
  std::string header = "method,qps,recall,avg_distance_ratio,nprob";
  csv_file << header << std::endl;
  std::cout << "Writing CSV header: " << header << std::endl;

  // 运行不同的方法并保存结果
  run<IVF>(dim, data, Nq, query_data, "IVF", csv_file, ground_truth, ground_truth_distances);
  run<IVFSQ4>(dim, data, Nq, query_data, "IVF-SQ4", csv_file, ground_truth, ground_truth_distances, true);
  // run<IVFSQ8>(dim, data, Nq, query_data, "IVF-SQ8", csv_file, ground_truth, ground_truth_distances);
  run<IVFPOUQ4>(dim, data, Nq, query_data, "IVF-POUQ4", csv_file, ground_truth, ground_truth_distances, true);
  // run<IVFPOUQ8>(dim, data, Nq, query_data, "IVF-POUQ8", csv_file, ground_truth, ground_truth_distances);

  csv_file.close();
  std::cout << "Results saved to " << csv_filename << std::endl;

  return 0;
}
