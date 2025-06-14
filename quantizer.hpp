#pragma once

#include <omp.h>

#include "clusterer.hpp"
#include "optimizer.hpp"
#include "simd/l2distance.hpp"

namespace pouq {

class Quantizer {
public:
  Quantizer() = default;

  explicit Quantizer(size_t dimension) : dimension_(dimension) {}

  void set_dimension(size_t dimension) { dimension_ = dimension; }

  size_t get_dimension() const { return dimension_; }

  void train(const float *data, size_t data_size) {
    codebook_      = new std::pair<float, float>[dimension_ * 16];
    encoded_codes_ = new uint8_t[data_size];

#pragma omp parallel for
    for (size_t dim = 0; dim < dimension_; dim++) {
      const auto   value_frequency_map = count_freq(data, data_size, dim);
      const auto   cluster_bounds      = clustering(16, value_frequency_map);
      const size_t codebook_offset     = dim * 16;

      for (size_t i = 0; i < cluster_bounds.size(); i++) {
        auto [lower_bound, upper_bound] = cluster_bounds[i];
        if (lower_bound < upper_bound) {
          auto data_start = std::lower_bound(value_frequency_map.begin(),
              value_frequency_map.end(),
              lower_bound,
              [](const std::pair<float, size_t> &value_freq, const float threshold) -> bool {
                return value_freq.first < threshold;
              });
          auto data_end   = std::upper_bound(value_frequency_map.begin(),
              value_frequency_map.end(),
              upper_bound,
              [](const float threshold, const std::pair<float, size_t> &value_freq) -> bool {
                return threshold < value_freq.first;
              });

          const auto [optimized_lower, optimized_upper] = optimize_quantization_range(15.0f, data_start, data_end);
          lower_bound                                   = optimized_lower;
          upper_bound                                   = optimized_upper;
        }
        if (lower_bound == upper_bound) {
          codebook_[codebook_offset + i] = {lower_bound, 1.0};
        } else {
          codebook_[codebook_offset + i] = {lower_bound, (upper_bound - lower_bound) / 15.0f};
        }
      }

      static_cast<std::vector<std::pair<float, size_t>>>(value_frequency_map).clear();
      for (size_t i = dim; i < data_size; i += dimension_) {
        const float  data_value       = data[i];
        const auto   cluster_it       = std::upper_bound(cluster_bounds.begin(),
            cluster_bounds.end(),
            data_value,
            [](const float value, const std::pair<float, float> &bound) -> bool { return value < bound.first; });
        const size_t cluster_index    = cluster_it - cluster_bounds.begin() - 1;
        auto [lower_bound, step_size] = codebook_[codebook_offset + cluster_index];
        const float normalized_value  = std::clamp((data_value - lower_bound) / step_size + 0.5f, 0.0f, 15.0f);
        encoded_codes_[i]             = cluster_index | static_cast<size_t>(normalized_value) << 4;
      }
    }
  }

  float l2distance(const float *data, size_t data_index) const {
    return simd::l2distance_simd(data, data_index, dimension_, codebook_, encoded_codes_);
  }

  ~Quantizer() {
    delete[] codebook_;
    delete[] encoded_codes_;
  }

private:
  size_t                   dimension_     = 0;
  std::pair<float, float> *codebook_      = nullptr;
  uint8_t                 *encoded_codes_ = nullptr;

  std::vector<std::pair<float, size_t>> count_freq(const float *data, size_t data_size, const size_t dim) const {
    std::vector<float> sorted_data;
    sorted_data.reserve(data_size / dimension_);
    for (size_t i = dim; i < data_size; i += dimension_) {
      sorted_data.push_back(data[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    float                                 current_value = sorted_data[0];
    size_t                                count         = 1;
    std::vector<std::pair<float, size_t>> value_frequency_map;
    value_frequency_map.reserve(sorted_data.size());
    for (size_t i = 1; i < sorted_data.size(); i++) {
      if (sorted_data[i] == current_value) {
        count++;
      } else {
        value_frequency_map.emplace_back(current_value, count);
        current_value = sorted_data[i];
        count         = 1;
      }
    }

    value_frequency_map.emplace_back(current_value, count);
    return value_frequency_map;
  }
};

}  // namespace pouq
