#pragma once

#include <algorithm>
#include <limits>
#include <vector>

namespace pouq {

struct Task {
  int j;
  int left;
  int right;
  int opt_left;
  int opt_right;
};

std::vector<std::pair<float, float>>
segment(int k, const std::vector<std::pair<float, int>>& data_freq_map) {
  const int size = data_freq_map.size();
  k = std::min(size, k);

  std::vector sum_count(size + 1, 0.0);
  for (int i = 1; i <= size; ++i) {
    sum_count[i] = sum_count[i - 1] + static_cast<double>(data_freq_map[i - 1].second);
  }

  std::vector prev_dp(size + 1, std::numeric_limits<double>::max());
  std::vector curr_dp(size + 1, std::numeric_limits<double>::max());
  std::vector prev_idx(size + 1, std::vector<int>(k + 1, 0));
  prev_dp[0] = 0.0;

  for (int j = 1; j <= k; ++j) {
    std::vector<Task> tasks{{j, j, size, 0, size - 1}};
    tasks.reserve(size);

    while (!tasks.empty()) {
      auto [j, l, r, opt_l, opt_r] = tasks.back();
      tasks.pop_back();
      if (l > r) {
        continue;
      }

      const int mid = (l + r) / 2;
      const int start = std::max(j - 1, opt_l);
      const int end = std::min(mid - 1, opt_r);
      double min_cost = std::numeric_limits<double>::max();
      int split_pos = 0;
      for (int m = start; m <= end; ++m) {
        const double width = static_cast<double>(data_freq_map[mid - 1].first) -
                             static_cast<double>(data_freq_map[m].first);
        const double count = sum_count[mid] - sum_count[m];
        const double cost = prev_dp[m] + width * width * count;
        if (cost < min_cost) {
          min_cost = cost;
          split_pos = m;
        }
      }

      curr_dp[mid] = min_cost;
      prev_idx[mid][j] = split_pos;
      if (l < r) {
        tasks.push_back({j, mid + 1, r, split_pos, opt_r});
        tasks.push_back({j, l, mid - 1, opt_l, split_pos});
      }
    }

    std::swap(prev_dp, curr_dp);
    std::fill(curr_dp.begin(), curr_dp.end(), std::numeric_limits<double>::max());
  }

  std::vector<int> split_pos(k);
  int curr_pos = size;
  for (int j = k; j > 0; --j) {
    const int m = prev_idx[curr_pos][j];
    split_pos[j - 1] = m;
    curr_pos = m;
  }

  std::vector<std::pair<float, float>> bounds(k);
  for (int t = 0; t < k; ++t) {
    const int start = split_pos[t];
    const int end = t < k - 1 ? split_pos[t + 1] - 1 : size - 1;
    bounds[t] = {data_freq_map[start].first, data_freq_map[end].first};
  }

  return bounds;
}

}  // namespace pouq
