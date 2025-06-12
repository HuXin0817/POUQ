#pragma once

#include <omp.h>

#include "clusterer.hpp"
#include "optimizer.hpp"

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace pouq {

class POUQ8bit {
public:
  explicit POUQ8bit(size_t dimension) : dimension_(dimension) {}

  void train(const float *data, size_t data_size) {
    codebook_      = new std::pair<float, float>[dimension_ * (1 << 4)];
    encoded_codes_ = new uint8_t[data_size];

#pragma omp parallel for
    for (size_t group = 0; group < dimension_; group++) {
      const auto   value_frequency_map = count_freq(data, data_size, group);
      const auto   cluster_bounds      = clustering(1 << 4, value_frequency_map);
      const size_t codebook_offset     = group * (1 << 4);

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

          const auto [optimized_lower, optimized_upper] = optimize_quantization_range(div, data_start, data_end);
          lower_bound                                   = optimized_lower;
          upper_bound                                   = optimized_upper;
        }
        if (lower_bound == upper_bound) {
          codebook_[codebook_offset + i] = {lower_bound, 1.0};
        } else {
          codebook_[codebook_offset + i] = {lower_bound, (upper_bound - lower_bound) / div};
        }
      }

      static_cast<std::vector<std::pair<float, size_t>>>(value_frequency_map).clear();
      for (size_t i = group; i < data_size; i += dimension_) {
        const float  data_value      = data[i];
        const auto   cluster_it      = std::upper_bound(cluster_bounds.begin(),
            cluster_bounds.end(),
            data_value,
            [](const float value, const std::pair<float, float> &bound) -> bool { return value < bound.first; });
        const size_t cluster_index   = cluster_it - cluster_bounds.begin() - 1;
        auto [lower_bound, scale]    = codebook_[codebook_offset + cluster_index];
        const float normalized_value = std::clamp((data_value - lower_bound) / scale + 0.5f, 0.0f, div);
        set(encoded_codes_, 2 * i, cluster_index);
        set(encoded_codes_, 2 * i + 1, normalized_value);
      }
    }
  }

#if defined(__i386__) || defined(__x86_64__)
  float l2distance(const float *data, size_t data_index) const {
    float distance = 0.0f;

    const size_t simd_end   = (dimension_ / 8) * 8;
    __m256       sum_vector = _mm256_setzero_ps();

    for (size_t i = 0; i < simd_end; i += 8) {
      __m256 data_vector = _mm256_loadu_ps(&data[i]);

      __m256 decoded_vector;
      {
        alignas(32) float   decoded_values[8];
        alignas(32) uint8_t code_batch[8];

        for (size_t j = 0; j < 8; j++) {
          code_batch[j] = encoded_codes_[data_index + i + j];
        }

        __m256i code_vector   = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)code_batch));
        __m256i lower_nibbles = _mm256_and_si256(code_vector, _mm256_set1_epi32(0xF));
        __m256i upper_nibbles = _mm256_and_si256(_mm256_srli_epi32(code_vector, 4), _mm256_set1_epi32(0xF));

        alignas(32) int32_t lower_indices[8];
        alignas(32) int32_t upper_values[8];
        _mm256_store_si256((__m256i *)lower_indices, lower_nibbles);
        _mm256_store_si256((__m256i *)upper_values, upper_nibbles);

        for (size_t j = 0; j < 8; j++) {
          const size_t codebook_index = lower_indices[j] + (i + j) * (1 << 4);
          auto [lower_bound, scale]   = codebook_[codebook_index];
          decoded_values[j]           = lower_bound + scale * static_cast<float>(upper_values[j]);
        }

        decoded_vector = _mm256_load_ps(decoded_values);
      }

      __m256 diff_vector = _mm256_sub_ps(data_vector, decoded_vector);
      sum_vector         = _mm256_fmadd_ps(diff_vector, diff_vector, sum_vector);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum_vector, 1);
    __m128 sum_low  = _mm256_castps256_ps128(sum_vector);
    __m128 sum_128  = _mm_add_ps(sum_high, sum_low);

    sum_128  = _mm_hadd_ps(sum_128, sum_128);
    sum_128  = _mm_hadd_ps(sum_128, sum_128);
    distance = _mm_cvtss_f32(sum_128);

    for (size_t i = simd_end; i < dimension_; i++) {
      uint8_t encoded_value     = encoded_codes_[data_index + i];
      auto [lower_bound, scale] = codebook_[((encoded_value & 0xF) + i * (1 << 4))];
      float decoded_value       = lower_bound + scale * (encoded_value >> 4 & 0xF);
      float diff                = data[i] - decoded_value;
      distance += diff * diff;
    }

    return distance;
  }
#else

  float l2distance(const float *data, size_t data_index) const {
    float distance = 0.0f;
    for (size_t i = 0; i < dimension_; i++) {
      uint8_t encoded_value     = encoded_codes_[data_index + i];
      auto [lower_bound, scale] = codebook_[((encoded_value & 0xF) + i * (1 << 4))];
      float decoded_value       = lower_bound + scale * (encoded_value >> 4 & 0xF);
      float diff                = data[i] - decoded_value;
      distance += diff * diff;
    }
    return distance;
  }

#endif

  ~POUQ8bit() {
    delete[] codebook_;
    delete[] encoded_codes_;
  }

private:
  size_t                   dimension_     = 0;
  std::pair<float, float> *codebook_      = nullptr;
  uint8_t                 *encoded_codes_ = nullptr;

  static constexpr auto div = static_cast<float>((1 << 4) - 1);

  std::vector<std::pair<float, size_t>> count_freq(const float *data, size_t data_size, const size_t group) const {
    std::vector<float> sorted_data;
    sorted_data.reserve(data_size / dimension_);
    for (size_t i = group; i < data_size; i += dimension_) {
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

  void set(uint8_t *data, size_t index, size_t value) {
    value &= (1 << 4) - 1;
    const size_t byte_index      = index / 2;
    const size_t nibble_position = index % 2;

    if (nibble_position == 0) {
      data[byte_index] = data[byte_index] & 0xF0 | value;
    } else {
      data[byte_index] = data[byte_index] & 0x0F | value << 4;
    }
  }
};

}  // namespace pouq
