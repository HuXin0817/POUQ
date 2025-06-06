#pragma once

#include <algorithm>
#include <limits>
#include <vector>

namespace pouq {

class Clusterer {
public:
  virtual ~Clusterer() = default;

  virtual std::vector<std::pair<float, float>> operator()(size_t k,
      const std::vector<std::pair<float, size_t>>               &data_freq_map) {
    return {{data_freq_map.front().first, data_freq_map.back().first}};
  }
};

class KrangeClusterer final : public Clusterer {

  struct Task {
    size_t j;
    size_t left;
    size_t right;
    size_t opt_left;
    size_t opt_right;

    Task(const size_t j, const size_t left, const size_t right, const size_t opt_left, const size_t opt_right)
        : j(j), left(left), right(right), opt_left(opt_left), opt_right(opt_right) {}
  };

public:
  std::vector<std::pair<float, float>> operator()(size_t k,
      const std::vector<std::pair<float, size_t>>       &data_freq_map) override {
    const size_t size = data_freq_map.size();
    k                 = std::min(size, k);

    std::vector sum_count(size + 1, 0.0);
    for (size_t i = 1; i <= size; ++i) {
      sum_count[i] = sum_count[i - 1] + static_cast<double>(data_freq_map[i - 1].second);
    }

    std::vector prev_dp(size + 1, std::numeric_limits<double>::max());
    std::vector curr_dp(size + 1, std::numeric_limits<double>::max());
    std::vector prev_idx(size + 1, std::vector<size_t>(k + 1, 0));
    prev_dp[0] = 0.0;

    for (size_t j = 1; j <= k; ++j) {
      std::vector<Task> tasks{{j, j, size, 0, size - 1}};
      tasks.reserve(size);

      while (!tasks.empty()) {
        auto [j, l, r, opt_l, opt_r] = tasks.back();
        tasks.pop_back();
        if (l > r) {
          continue;
        }

        const size_t mid       = (l + r) / 2;
        const size_t start     = std::max(j - 1, opt_l);
        const size_t end       = std::min(mid - 1, opt_r);
        double       min_cost  = std::numeric_limits<double>::max();
        size_t       split_pos = 0;
        for (size_t m = start; m <= end; ++m) {
          const double width =
              static_cast<double>(data_freq_map[mid - 1].first) - static_cast<double>(data_freq_map[m].first);
          const double count = sum_count[mid] - sum_count[m];
          const double cost  = prev_dp[m] + width * width * count;
          if (cost < min_cost) {
            min_cost  = cost;
            split_pos = m;
          }
        }

        curr_dp[mid]     = min_cost;
        prev_idx[mid][j] = split_pos;
        if (l < r) {
          tasks.emplace_back(j, mid + 1, r, split_pos, opt_r);
          tasks.emplace_back(j, l, mid - 1, opt_l, split_pos);
        }
      }

      std::swap(prev_dp, curr_dp);
      std::fill(curr_dp.begin(), curr_dp.end(), std::numeric_limits<double>::max());
    }

    std::vector<size_t> split_pos(k);
    size_t              curr_pos = size;
    for (size_t j = k; j > 0; --j) {
      const size_t m   = prev_idx[curr_pos][j];
      split_pos[j - 1] = m;
      curr_pos         = m;
    }

    std::vector<std::pair<float, float>> bounds(k);
    for (size_t t = 0; t < k; ++t) {
      const size_t start = split_pos[t];
      const size_t end   = t < k - 1 ? split_pos[t + 1] - 1 : size - 1;
      bounds[t]          = {data_freq_map[start].first, data_freq_map[end].first};
    }

    return bounds;
  }
};

class KmeansClusterer final : public Clusterer {

  struct Task {
    size_t j;
    size_t left;
    size_t right;
    size_t opt_left;
    size_t opt_right;

    Task(const size_t j, const size_t left, const size_t right, const size_t opt_left, const size_t opt_right)
        : j(j), left(left), right(right), opt_left(opt_left), opt_right(opt_right) {}
  };

public:
  std::vector<std::pair<float, float>> operator()(size_t k,
      const std::vector<std::pair<float, size_t>>       &data_freq_map) override {
    const size_t size = data_freq_map.size();
    k                 = std::min(size, k);

    std::vector sum_count(size + 1, 0.0);
    std::vector sum_weighted(size + 1, 0.0);
    std::vector sum_squared_weighted(size + 1, 0.0);

    for (size_t i = 1; i <= size; ++i) {
      const double value = data_freq_map[i - 1].first;
      const double count = data_freq_map[i - 1].second;

      sum_count[i]            = sum_count[i - 1] + count;
      sum_weighted[i]         = sum_weighted[i - 1] + value * count;
      sum_squared_weighted[i] = sum_squared_weighted[i - 1] + value * value * count;
    }

    std::vector prev_dp(size + 1, std::numeric_limits<double>::max());
    std::vector curr_dp(size + 1, std::numeric_limits<double>::max());
    std::vector prev_idx(size + 1, std::vector<size_t>(k + 1, 0));
    prev_dp[0] = 0.0;

    for (size_t j = 1; j <= k; ++j) {
      std::vector<Task> tasks{{j, j, size, 0, size - 1}};
      tasks.reserve(size);

      while (!tasks.empty()) {
        auto [j, l, r, opt_l, opt_r] = tasks.back();
        tasks.pop_back();
        if (l > r) {
          continue;
        }

        const size_t mid       = (l + r) / 2;
        const size_t start     = std::max(j - 1, opt_l);
        const size_t end       = std::min(mid - 1, opt_r);
        double       min_cost  = std::numeric_limits<double>::max();
        size_t       split_pos = 0;

        for (size_t m = start; m <= end; ++m) {
          const double total_count = sum_count[mid] - sum_count[m];

          if (total_count > 0) {
            const double total_weighted         = sum_weighted[mid] - sum_weighted[m];
            const double total_squared_weighted = sum_squared_weighted[mid] - sum_squared_weighted[m];
            const double centroid               = total_weighted / total_count;
            const double cluster_cost           = total_squared_weighted - centroid * total_weighted;
            const double cost                   = prev_dp[m] + cluster_cost;
            if (cost < min_cost) {
              min_cost  = cost;
              split_pos = m;
            }
          }
        }

        curr_dp[mid]     = min_cost;
        prev_idx[mid][j] = split_pos;
        if (l < r) {
          tasks.emplace_back(j, mid + 1, r, split_pos, opt_r);
          tasks.emplace_back(j, l, mid - 1, opt_l, split_pos);
        }
      }

      std::swap(prev_dp, curr_dp);
      std::fill(curr_dp.begin(), curr_dp.end(), std::numeric_limits<double>::max());
    }

    std::vector<size_t> split_pos(k);
    size_t              curr_pos = size;
    for (size_t j = k; j > 0; --j) {
      const size_t m   = prev_idx[curr_pos][j];
      split_pos[j - 1] = m;
      curr_pos         = m;
    }

    std::vector<std::pair<float, float>> bounds(k);
    for (size_t t = 0; t < k; ++t) {
      const size_t start = split_pos[t];
      const size_t end   = t < k - 1 ? split_pos[t + 1] - 1 : size - 1;
      bounds[t]          = {data_freq_map[start].first, data_freq_map[end].first};
    }

    return bounds;
  }
};

}  // namespace pouq
