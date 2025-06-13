#pragma once

#include <vector>

namespace pouq::simd {

inline std::pair<float, size_t> dp_cost_simd(size_t j,
    size_t                                          mid,
    size_t                                          opt_l,
    size_t                                          opt_r,
    const std::pair<float, size_t>                 *data,
    const float                                    *cnt,
    const float                                    *dp) {
  const size_t start     = std::max(j - 1, opt_l);
  const size_t end       = std::min(mid - 1, opt_r);
  float        min_cost  = std::numeric_limits<float>::max();
  size_t       split_pos = 0;
  for (size_t m = start; m <= end; ++m) {
    const float w    = static_cast<float>(data[mid - 1].first) - static_cast<float>(data[m].first);
    const float c    = cnt[mid] - cnt[m];
    const float cost = dp[m] + w * w * c;
    if (cost < min_cost) {
      min_cost  = cost;
      split_pos = m;
    }
  }
  return {min_cost, split_pos};
}

};  // namespace pouq::simd