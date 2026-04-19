#pragma once

#include <vector>
#include <cstdint>

namespace pouq::cluster {

class Clusterer {
 public:
  Clusterer(uint32_t n_clusters) : n_clusters_(n_clusters) {}

  ~Clusterer() = default;

  std::pair<std::vector<float>, std::vector<float>> Split(const std::vector<float>& data);

 private:
  uint32_t n_clusters_;
};

}  // namespace pouq::cluster