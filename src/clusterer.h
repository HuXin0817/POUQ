#pragma once

#include <vector>

namespace pouq {

class Clusterer {
 public:
  Clusterer(int n_clusters) : n_clusters_(n_clusters) {}

  ~Clusterer() = default;

  std::pair<std::vector<float>, std::vector<float>> Split(const std::vector<float>& data);

 private:
  int n_clusters_;
};

}  // namespace pouq