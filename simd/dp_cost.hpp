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

inline float quantization_loss_simd(const float                  d,
    float                                                        l,
    float                                                        s,
    const std::vector<std::pair<float, size_t>>::const_iterator &begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &end) {
  s          = std::max(s, 1e-8f);
  float loss = 0.0f;

  for (auto it = begin; it != end; ++it) {
    const auto &[v, c] = *it;
    const float rc     = (v - l) / s;
    float       qc     = 0.0f;
    if (v > l) {
      qc = std::round(rc);
      if (qc > d) {
        qc = d;
      }
    }
    const float err = rc - qc;
    loss += err * err * static_cast<float>(c);
  }

  return loss * s * s;
}

inline float l2distance_simd(const float *data,
    size_t                                index,
    size_t                                dim,
    const std::pair<float, float>        *code,
    const uint8_t                        *encode) {
  float dis = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    const uint8_t c   = encode[index + i];
    const auto [l, s] = code[((c & 0xF) + i * 16)];
    const float v     = l + s * (c >> 4 & 0xF);
    const float diff  = data[i] - v;
    dis += diff * diff;
  }
  return dis;
}

};  // namespace pouq::simd