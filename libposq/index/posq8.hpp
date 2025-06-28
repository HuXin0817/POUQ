#pragma once

#include <omp.h>

#include "../bitmap.hpp"
#include "../clusterer.hpp"
#include "../optimizer.hpp"

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>  // AVX2指令集头文件
#endif

namespace posq {

class POSQ8 {
public:
  explicit POSQ8(size_t dim) : dim_(dim) {}

  void train(const float *data, size_t size) {
    // size_        = size;
    codebook_ = new std::pair<float, float>[dim_ * (1 << 4)];
    codes_    = new uint8_t[size];

#pragma omp parallel for
    for (size_t group = 0; group < dim_; group++) {
      const auto data_freq_map = count_freq(data, size, group);
      const auto bounds        = clusterer(1 << 4, data_freq_map);
      const auto offset        = group * (1 << 4);

      for (size_t i = 0; i < bounds.size(); i++) {
        auto [lower, upper] = bounds[i];
        if (lower < upper) {
          auto data_start = std::lower_bound(data_freq_map.begin(),
              data_freq_map.end(),
              lower,
              [](const std::pair<float, size_t> &lhs, const float rhs) -> bool { return lhs.first < rhs; });
          auto data_end   = std::upper_bound(data_freq_map.begin(),
              data_freq_map.end(),
              upper,
              [](const float rhs, const std::pair<float, size_t> &lhs) -> bool { return rhs < lhs.first; });

          const auto [opt_lower, opt_upper] = optimizer(div, data_start, data_end);
          lower                             = opt_lower;
          upper                             = opt_upper;
        }
        if (lower == upper) {
          codebook_[offset + i] = {lower, 1.0};
        } else {
          codebook_[offset + i] = {lower, (upper - lower) / div};
        }
      }

      static_cast<std::vector<std::pair<float, size_t>>>(data_freq_map).clear();
      for (size_t i = group; i < size; i += dim_) {
        const float d  = data[i];
        const auto  it = std::upper_bound(
            bounds.begin(), bounds.end(), d, [](const float rhs, const std::pair<float, float> &lhs) -> bool {
              return rhs < lhs.first;
            });
        const size_t c = it - bounds.begin() - 1;
        set(codes_, 2 * i, c);
        auto [lb, s]  = codebook_[offset + c];
        const float x = std::clamp((d - lb) / s + 0.5f, 0.0f, div);
        set(codes_, 2 * i + 1, x);
      }
    }
  }

#if defined(__i386__) || defined(__x86_64__)
  float l2distance(const float *data, size_t n) const {
    float dis = 0.0f;

    // AVX2优化版本 - 一次处理8个float
    const size_t simd_end = (dim_ / 8) * 8;
    __m256       sum_vec  = _mm256_setzero_ps();

    for (size_t i = 0; i < simd_end; i += 8) {
      // 加载8个输入数据
      __m256 data_vec = _mm256_loadu_ps(&data[i]);

      // 批量解码8个量化值 - 关键优化点
      __m256 decoded_vec;
      {
        // 预先计算codebook索引和值
        alignas(32) float   decoded[8];
        alignas(32) uint8_t codes_batch[8];

        // 批量加载codes
        for (size_t j = 0; j < 8; j++) {
          codes_batch[j] = codes_[n + i + j];
        }

        // 向量化解码过程
        __m256i codes_vec   = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)codes_batch));
        __m256i lower_4bits = _mm256_and_si256(codes_vec, _mm256_set1_epi32(0xF));
        __m256i upper_4bits = _mm256_and_si256(_mm256_srli_epi32(codes_vec, 4), _mm256_set1_epi32(0xF));

        // 批量查找codebook值
        alignas(32) int32_t lower_indices[8];
        alignas(32) int32_t upper_values[8];
        _mm256_store_si256((__m256i *)lower_indices, lower_4bits);
        _mm256_store_si256((__m256i *)upper_values, upper_4bits);

        for (size_t j = 0; j < 8; j++) {
          const size_t codebook_idx = lower_indices[j] + (i + j) * (1 << 4);
          auto [lb, s]              = codebook_[codebook_idx];
          decoded[j]                = lb + s * static_cast<float>(upper_values[j]);
        }

        decoded_vec = _mm256_load_ps(decoded);
      }

      // 计算差值
      __m256 diff_vec = _mm256_sub_ps(data_vec, decoded_vec);

      // 计算平方并累加 - 使用FMA指令优化
      sum_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_vec);
    }

    // 优化的水平求和
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low  = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128  = _mm_add_ps(sum_high, sum_low);

    // 使用更高效的水平求和
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    dis     = _mm_cvtss_f32(sum_128);

    // 处理剩余的元素（标量处理）
    for (size_t i = simd_end; i < dim_; i++) {
      uint8_t v    = codes_[n + i];
      auto [lb, s] = codebook_[((v & 0xF) + i * (1 << 4))];
      float decode = lb + s * (v >> 4 & 0xF);
      float diff   = data[i] - decode;
      dis += diff * diff;
    }

    return dis;
  }
#else

  float l2distance(const float *data, size_t n) const {
    float dis = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
      uint8_t v    = codes_[n + i];
      auto [lb, s] = codebook_[((v & 0xF) + i * (1 << 4))];
      float decode = lb + s * (v >> 4 & 0xF);
      float diff   = data[i] - decode;
      dis += diff * diff;
    }
    return dis;
  }

#endif

  // size_t size() const { return size_; }

  ~POSQ8() {
    delete[] codebook_;
    delete[] codes_;
  }

private:
  // size_t   size_        = 0;
  size_t dim_ = 0;
  // float *codebook_ = nullptr;
  // float *codebook_   = nullptr;
  std::pair<float, float> *codebook_ = nullptr;
  // uint8_t *cid_         = nullptr;
  // uint8_t *code_        = nullptr;
  uint8_t *codes_ = nullptr;

  static inline KrangeClusterer clusterer;
  static inline PSOptimizer     optimizer;

  static constexpr auto div = static_cast<float>((1 << 4) - 1);

  std::vector<std::pair<float, size_t>> count_freq(const float *data, size_t size, const size_t group) const {
    std::vector<float> sorted_data;
    sorted_data.reserve(size / dim_);
    for (size_t i = group; i < size; i += dim_) {
      sorted_data.push_back(data[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    float                                 curr_value = sorted_data[0];
    size_t                                count      = 1;
    std::vector<std::pair<float, size_t>> data_freq_map;
    data_freq_map.reserve(sorted_data.size());
    for (size_t i = 1; i < sorted_data.size(); i++) {
      if (sorted_data[i] == curr_value) {
        count++;
      } else {
        data_freq_map.emplace_back(curr_value, count);
        curr_value = sorted_data[i];
        count      = 1;
      }
    }

    data_freq_map.emplace_back(curr_value, count);
    return data_freq_map;
  }

  // float at(size_t i) const {
  //   const auto   v      = codes_[i];
  //   const size_t group  = i % dim_;
  //   const size_t offset = 2 * ((v & 0xF) + group * (1 << 4));
  //   return codebook_[offset] + codebook_[offset + 1] * static_cast<float>(v >> 4 & 0xF);
  // }

  void set(uint8_t *data, size_t index, size_t n) {
    n &= (1 << 4) - 1;
    const size_t pos = index * 4;
    for (size_t bit = 0; bit < 4; ++bit) {
      const size_t i      = (pos + bit) / 8;
      const size_t offset = (pos + bit) % 8;
      if (n & 1 << bit) {
        data[i] |= 1 << offset;
      } else {
        data[i] &= ~(1 << offset);
      }
    }
  }

  // std::pair<size_t, size_t> get_pair(const uint8_t *data, size_t index) const {
  //   const size_t pos = index * 4;
  //   const size_t byte_idx = pos / 8;
  //   const size_t bit_offset = pos % 8;
  //   const uint8_t byte_val = data[byte_idx];
  //
  //   const size_t b1 = (byte_val >> bit_offset) & 0xF;
  //   const size_t b2 = (byte_val >> (bit_offset + 4)) & 0xF;
  //   return {b1, b2};
  // }
};

}  // namespace posq