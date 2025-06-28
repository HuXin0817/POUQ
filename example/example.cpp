#include "../libposq/ivf.hpp"
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

  IvfIndex index(16, dim);
  index.train(data.data(), data.size());

  float sum_recall = 0.0f;

  auto cmp = [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b) { return a.second > b.second; };
  for (size_t i = 0; i < Nq; i++) {

    std::unordered_set<size_t> real_idx;
    const auto                 q = query_data.data() + i * dim;
    {
      std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t, float>>, decltype(cmp)> pq(cmp);
      for (size_t p = 0; p < data.size() / dim; p++) {
        pq.emplace(p, l2distance(q, data.data() + p * dim, dim));
      }
      for (size_t k = 0; k < 10; k++) {
        real_idx.insert(pq.top().first);
        pq.pop();
      }
    }

    constexpr auto topk = 5;
    auto           ret  = index.search(q, topk, 4);

    size_t finded = 0;
    for (auto idx : ret) {
      if (real_idx.find(idx) != real_idx.end()) {
        finded += 1;
      }
    }

    sum_recall += static_cast<float>(finded) / static_cast<float>(topk);
  }

  sum_recall /= Nq;
  std::cout << sum_recall << std::endl;
  return 0;
}
