#pragma once

#include <algorithm>
#include <limits>
#include <map>
#include <vector>

namespace pouq {

std::pair<std::vector<float>, std::vector<float>> krange(const uint64_t cluster_num,
    const std::vector<std::pair<float, uint64_t>>                      &data_freq_map) {
  std::vector<float>    data_values;
  std::vector<uint64_t> data_counts;
  data_values.reserve(data_freq_map.size());
  data_counts.reserve(data_freq_map.size());
  for (const auto &[value, count] : data_freq_map) {
    data_values.push_back(value);
    data_counts.push_back(count);
  }

  if (data_values.size() < cluster_num) {
    return {data_values, data_values};
  }

  const uint64_t data_size = data_values.size();
  std::vector    cum_sum_counts(data_size + 1, 0.0f);
  for (uint64_t i = 1; i <= data_size; ++i) {
    cum_sum_counts[i] = cum_sum_counts[i - 1] + static_cast<float>(data_counts[i - 1]);
  }

  std::vector prev_dp(data_size + 1, std::numeric_limits<float>::infinity());
  std::vector curr_dp(data_size + 1, std::numeric_limits<float>::infinity());
  std::vector prev_indices(data_size + 1, std::vector<uint64_t>(cluster_num + 1, 0));
  prev_dp[0] = 0.0f;

  struct Task {
    uint64_t current_cluster;
    uint64_t left_index;
    uint64_t right_index;
    uint64_t opt_left;
    uint64_t opt_right;
  };

  for (uint64_t j = 1; j <= cluster_num; ++j) {
    std::vector<Task> task_stack;
    task_stack.reserve(data_size);
    task_stack.push_back({j, j, data_size, 0, data_size - 1});

    while (!task_stack.empty()) {
      auto [curr_j, l, r, opt_l, opt_r] = task_stack.back();
      task_stack.pop_back();
      if (l > r) {
        continue;
      }

      const uint64_t mid_pos         = (l + r) / 2;
      float          best_total_cost = std::numeric_limits<float>::infinity();
      uint64_t       best_split_pos  = 0;

      const uint64_t search_start = std::max(curr_j - 1, opt_l);
      const uint64_t search_end   = std::min(mid_pos - 1, opt_r);

      for (uint64_t m = search_start; m <= search_end; ++m) {
        const float range_width  = data_values[mid_pos - 1] - data_values[m];
        const float point_count  = cum_sum_counts[mid_pos] - cum_sum_counts[m];
        const float cluster_cost = range_width * range_width * point_count;
        const float total_cost   = prev_dp[m] + cluster_cost;
        if (total_cost < best_total_cost) {
          best_total_cost = total_cost;
          best_split_pos  = m;
        }
      }

      curr_dp[mid_pos]              = best_total_cost;
      prev_indices[mid_pos][curr_j] = best_split_pos;

      if (l < r) {
        task_stack.push_back({curr_j, mid_pos + 1, r, best_split_pos, opt_r});
        task_stack.push_back({curr_j, l, mid_pos - 1, opt_l, best_split_pos});
      }
    }

    std::swap(prev_dp, curr_dp);
    std::fill(curr_dp.begin(), curr_dp.end(), std::numeric_limits<float>::infinity());
  }

  std::vector<uint64_t> split_positions;
  uint64_t              current_pos = data_size, current_j = cluster_num;
  while (current_j > 0) {
    const uint64_t m = prev_indices[current_pos][current_j];
    split_positions.push_back(m);
    current_pos = m;
    current_j--;
  }
  std::reverse(split_positions.begin(), split_positions.end());

  std::vector<float> lower_bounds, upper_bounds;
  for (uint64_t t = 0; t < cluster_num; ++t) {
    const uint64_t start_index = split_positions[t];
    const uint64_t end_index   = (t < cluster_num - 1) ? split_positions[t + 1] - 1 : data_size - 1;
    lower_bounds.push_back(data_values[start_index]);
    upper_bounds.push_back(data_values[end_index]);
  }

  return {lower_bounds, upper_bounds};
}

}  // namespace pouq
