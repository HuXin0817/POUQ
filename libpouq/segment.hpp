#pragma once

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <utility>
#include <vector>

namespace pouq::segment {

struct Task {
  int j;
  int left;
  int right;
  int opt_left;
  int opt_right;
};

static std::vector<std::pair<float, float>>
segment(int k, const float* data_map, const int* freq_map, int size, bool do_count_freq) {
  assert(k > 0);
  assert(size > 0);
  assert(data_map != nullptr);
  if (do_count_freq) {
    assert(freq_map != nullptr);
  }

  k = std::min(size, k);

  std::vector<int> sum_count;
  if (do_count_freq) {
    sum_count.resize(size + 1, 0);
    for (int i = 1; i <= size; ++i) {
      sum_count[i] = sum_count[i - 1] + freq_map[i - 1];
    }
  }

  std::vector<float> prev_dp(size + 1, FLT_MAX);
  std::vector<float> curr_dp(size + 1, FLT_MAX);
  std::vector<std::vector<int>> prev_idx(size + 1, std::vector<int>(k + 1, 0));
  prev_dp[0] = 0.0f;

  for (int j = 1; j <= k; ++j) {
    std::vector<Task> tasks{{j, j, size, 0, size - 1}};
    tasks.reserve(size);

    while (!tasks.empty()) {
      auto [j, l, r, opt_l, opt_r] = tasks.back();
      tasks.pop_back();
      if (l > r) {
        continue;
      }

      int mid = (l + r) / 2;
      int start = std::max(j - 1, opt_l);
      int end = std::min(mid - 1, opt_r);
      float min_cost = FLT_MAX;
      int split_pos = 0;
      for (int m = start; m <= end; ++m) {
        float width = data_map[mid - 1] - data_map[m];
        float cost = prev_dp[m];
        if (do_count_freq) {
          int count = sum_count[mid] - sum_count[m];
          cost += width * width * static_cast<float>(count);
        } else {
          cost += width * width;
        }
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
    std::fill(curr_dp.begin(), curr_dp.end(), FLT_MAX);
  }

  std::vector<int> split_pos(k);
  int curr_pos = size;
  for (int j = k; j > 0; --j) {
    int m = prev_idx[curr_pos][j];
    split_pos[j - 1] = m;
    curr_pos = m;
  }

  std::vector<std::pair<float, float>> bounds(k);
  for (int t = 0; t < k; ++t) {
    int start = split_pos[t];
    int end = t < k - 1 ? split_pos[t + 1] - 1 : size - 1;
    bounds[t] = {data_map[start], data_map[end]};
  }

  return bounds;
}

}  // namespace pouq::segment