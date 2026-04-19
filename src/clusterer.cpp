#include "clusterer.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>
#include <vector>

namespace pouq::cluster {

std::pair<std::vector<float>, std::vector<float>> Clusterer::Split(const std::vector<float>& data) {
  uint32_t n = data.size();
  assert(n > 0);

  uint32_t k = std::min(n, n_clusters_);

  std::vector<float> dp_prev(n), dp_curr(n);
  std::vector<std::vector<uint32_t>> opt(k + 1, std::vector<uint32_t>(n));

  for (uint32_t i = 0; i < n; ++i) {
    float diff = data[i] - data[0];
    dp_prev[i] = diff * diff * (i + 1);
  }

  for (uint32_t p = 2; p <= k; ++p) {
    for (uint32_t i = p - 1; i < n; ++i) {
      uint32_t low = (i == p - 1) ? p - 2 : opt[p][i - 1];
      uint32_t high = i - 1;
      uint32_t best_j = low;
      float min_val = std::numeric_limits<float>::max();

      for (uint32_t j = low; j <= high; ++j) {
        float diff = data[i] - data[j + 1];
        float count = i - j;
        float val = dp_prev[j] + diff * diff * count;

        if (val < min_val) {
          min_val = val;
          best_j = j;
        }
      }
      dp_curr[i] = min_val;
      opt[p][i] = best_j;
    }
    std::swap(dp_prev, dp_curr);
  }

  std::vector<float> lefts(k), rights(k);
  uint32_t cur_i = n - 1;
  for (uint32_t p = k; p >= 1; --p) {
    uint32_t split_j = (p == 1) ? -1 : opt[p][cur_i];

    uint32_t l_idx = split_j + 1;
    uint32_t r_idx = cur_i;
    lefts[p - 1] = data[l_idx];
    rights[p - 1] = data[r_idx];

    cur_i = split_j;
  }

  return {lefts, rights};
}

}  // namespace pouq::cluster