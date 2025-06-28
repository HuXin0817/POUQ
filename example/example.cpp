#include "../libposq/index/ivf.hpp"
#include <chrono>
#include <iomanip>
#include <string>
#include <unordered_set>

std::vector<float> generate_vector(size_t size) {
  // std::random_device                    rd;
  std::mt19937                          gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  std::vector<float> result(size);

#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    result[i] = dis(gen);
  }

  return result;
}

int main(int argc, char *argv[]) {
  const std::string dataset_name = argv[1];
  // size_t dim        = 100;
  // auto   data       = generate_vector(dim * 50000);
  // auto   query_data = generate_vector(dim * 10);

  const auto [data, dim] = read_fvecs("../data/" + dataset_name + "/" + dataset_name + "_base.fvecs");
  auto [query_data, _]   = read_fvecs("../data/" + dataset_name + "/" + dataset_name + "_query.fvecs");
  query_data             = std::vector(query_data.begin(), query_data.begin() + dim * 100);
  auto Nq                = query_data.size() / dim;

  std::cout << "Data shape: (" << data.size() / dim << ", " << dim << ")" << std::endl;
  std::cout << "Query shape: (" << Nq << ", " << dim << ")" << std::endl;

  IvfIndex index(100, dim);
  index.train(data.data(), data.size());

  constexpr auto topk = 10;

  // 计算ground truth
  std::cout << "Computing ground truth using brute force search..." << std::endl;
  std::vector<std::vector<size_t>> ground_truth(Nq);
  auto cmp = [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b) { return a.second > b.second; };

  for (size_t i = 0; i < Nq; i++) {
    std::unordered_set<size_t> real_idx;
    const auto                 q = query_data.data() + i * dim;
    {
      std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t, float>>, decltype(cmp)> pq(cmp);
      for (size_t p = 0; p < data.size() / dim; p++) {
        pq.emplace(p, l2distance(q, data.data() + p * dim, dim));
      }
      for (size_t k = 0; k < topk; k++) {
        ground_truth[i].push_back(pq.top().first);
        pq.pop();
      }
    }
  }

  // QPS测试
  std::cout << "\n=== Testing POSQ IVF Index ===" << std::endl;

  // 测试不同的nprobe值
  std::vector<size_t> nprobe_values = {1, 5, 10, 20, 50};

  std::cout << std::left << std::setw(15) << "nprobe" << std::setw(15) << "QPS" << std::setw(15) << "Recall@10"
            << std::setw(15) << "Time(s)" << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  for (auto nprobe : nprobe_values) {
    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行搜索
    std::vector<std::vector<size_t>> search_results(Nq);
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
        if (gt_set.find(search_results[i][j]) != gt_set.end()) {
          found++;
        }
      }
      sum_recall += static_cast<float>(found) / static_cast<float>(topk);
    }
    float avg_recall = sum_recall / Nq;

    // 输出结果
    std::cout << std::left << std::setw(15) << nprobe << std::setw(15) << std::fixed << std::setprecision(2) << qps
              << std::setw(15) << std::fixed << std::setprecision(4) << avg_recall << std::setw(15) << std::fixed
              << std::setprecision(4) << total_time << std::endl;
  }

  return 0;
}
