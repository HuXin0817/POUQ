#pragma once

#include <vector>

namespace pouq::simd {

inline std::pair<float, size_t> dp_cost_simd(size_t j,
    size_t                                          mid,
    size_t                                          opt_l,
    size_t                                          opt_r,
    const std::pair<float, size_t>                 *data_freq_map,
    const float                                    *sum_count,
    const float                                    *prev_dp) {
  const size_t start     = std::max(j - 1, opt_l);
  const size_t end       = std::min(mid - 1, opt_r);
  float        min_cost  = std::numeric_limits<float>::max();
  size_t       split_pos = 0;
  for (size_t m = start; m <= end; ++m) {
    const float width = static_cast<float>(data_freq_map[mid - 1].first) - static_cast<float>(data_freq_map[m].first);
    const float count = sum_count[mid] - sum_count[m];
    const float cost  = prev_dp[m] + width * width * count;
    if (cost < min_cost) {
      min_cost  = cost;
      split_pos = m;
    }
  }
  return {min_cost, split_pos};
}

inline float quantization_loss_simd(const float                  division_count,
    float                                                        cluster_lower_bound,
    float                                                        step_size,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_end) {
  step_size        = std::max(step_size, 1e-8f);
  float total_loss = 0.0f;

  for (auto it = data_begin; it != data_end; ++it) {
    const auto &[data_value, point_count] = *it;
    const float real_quantized_code       = (data_value - cluster_lower_bound) / step_size;
    float       quantized_code            = 0.0f;

    if (data_value > cluster_lower_bound) {
      quantized_code = std::round(real_quantized_code);
      if (quantized_code > division_count) {
        quantized_code = division_count;
      }
    }

    const float quantization_error = real_quantized_code - quantized_code;
    total_loss += quantization_error * quantization_error * static_cast<float>(point_count);
  }

  return total_loss * step_size * step_size;
}

inline float l2distance_simd(const float *data,
    size_t                                data_index,
    size_t                                dimension,
    const std::pair<float, float>        *codebook,
    const uint8_t                        *encoded_codes) {
  float distance = 0.0f;
  for (size_t i = 0; i < dimension; i++) {
    const uint8_t encoded_value         = encoded_codes[data_index + i];
    const auto [lower_bound, step_size] = codebook[((encoded_value & 0xF) + i * 16)];
    const float decoded_value           = lower_bound + step_size * (encoded_value >> 4 & 0xF);
    const float diff                    = data[i] - decoded_value;
    distance += diff * diff;
  }
  return distance;
}

};  // namespace pouq::simd