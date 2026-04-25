#include "clusterer.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>
#include <vector>

namespace pouq::cluster {

std::pair<std::vector<float>, std::vector<float>> Clusterer::Split(const std::vector<float>& data) {
  int n = data.size();
  assert(n > 0);

  int k = std::min(n, n_clusters_);

  std::vector<float> dp_prev(n), dp_curr(n);
  std::vector<std::vector<int>> opt(k + 1, std::vector<int>(n));

  for (int i = 0; i < n; ++i) {
    float diff = data[i] - data[0];
    dp_prev[i] = diff * diff * (i + 1);
  }

  for (int p = 2; p <= k; ++p) {
    for (int i = p - 1; i < n; ++i) {
      int low = (i == p - 1) ? p - 2 : opt[p][i - 1];
      int high = i - 1;
      int best_j = low;
      float min_val = std::numeric_limits<float>::max();

      for (int j = low; j <= high; ++j) {
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
  int cur_i = n - 1;
  for (int p = k; p >= 1; --p) {
    int split_j = (p == 1) ? -1 : opt[p][cur_i];

    int l_idx = split_j + 1;
    int r_idx = cur_i;
    lefts[p - 1] = data[l_idx];
    rights[p - 1] = data[r_idx];

    cur_i = split_j;
  }

  return {lefts, rights};
}

}  // namespace pouq::cluster