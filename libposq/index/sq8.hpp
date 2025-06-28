#pragma once

#include <omp.h>

#include "../clusterer.hpp"
#include "../optimizer.hpp"

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>  // AVX2指令集头文件
#endif

namespace posq {

class SQ8 {
public:
  explicit SQ8(size_t dim) : dim_(dim) {}

  static constexpr auto div = static_cast<float>((1 << 8) - 1);

  void train(const float *data, size_t size) {
    step_size_   = new float[dim_];
    lower_bound_ = new float[dim_];
    code_        = new uint8_t[size];

#pragma omp parallel for
    for (size_t group = 0; group < dim_; group++) {
      const auto data_freq_map = count_freq(data, size, group);
      const auto bounds        = clusterer(1 << 0, data_freq_map);
      const auto offset        = group;

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
        lower_bound_[offset + i] = lower;
        if (lower == upper) {
          step_size_[offset + i] = 1.0;
        } else {
          step_size_[offset + i] = (upper - lower) / div;
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
        code_[i]       = std::clamp((d - lower_bound_[offset + c]) / step_size_[offset + c] + 0.5f, 0.0f, div);
      }
    }
  }

  // float operator[](size_t i) const {
  //   const size_t group  = i % dim_;
  //   const size_t offset = group;
  //   const size_t x      = code_[i];
  //   return lower_bound_[offset] + step_size_[offset] * static_cast<float>(x);
  // }

#if defined(__i386__) || defined(__x86_64__)

  float l2distance(const float *data, size_t n) const {
    float  dis = 0.0f;
    size_t i   = 0;

    // SIMD向量化处理，每次处理8个float
    const size_t simd_end = (dim_ / 8) * 8;
    __m256       sum_vec  = _mm256_setzero_ps();

    for (; i < simd_end; i += 8) {
      // 加载8个uint8_t值并转换为float
      __m128i codes_128   = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&code_[n + i]));
      __m256i codes_256   = _mm256_cvtepu8_epi32(codes_128);
      __m256  codes_float = _mm256_cvtepi32_ps(codes_256);

      // 右移4位并与0xF进行AND操作
      __m256i shifted  = _mm256_srli_epi32(_mm256_cvtps_epi32(codes_float), 4);
      __m256i masked   = _mm256_and_si256(shifted, _mm256_set1_epi32(0xF));
      __m256  v_values = _mm256_cvtepi32_ps(masked);

      // 加载lower_bound和step_size
      __m256 lb_vec = _mm256_loadu_ps(&lower_bound_[i]);
      __m256 s_vec  = _mm256_loadu_ps(&step_size_[i]);

      // 计算decode = lb + s * v_values
      __m256 decode_vec = _mm256_fmadd_ps(s_vec, v_values, lb_vec);

      // 加载data并计算差值
      __m256 data_vec = _mm256_loadu_ps(&data[i]);
      __m256 diff_vec = _mm256_sub_ps(data_vec, decode_vec);

      // 计算平方并累加
      sum_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_vec);
    }

    // 水平求和SIMD结果
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low  = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128  = _mm_add_ps(sum_high, sum_low);

    // 继续水平求和
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    dis     = _mm_cvtss_f32(sum_128);

    // 处理剩余的元素（标量处理）
    for (; i < dim_; i++) {
      uint8_t v  = code_[n + i];
      auto    lb = lower_bound_[i];
      auto    s  = step_size_[i];

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
      uint8_t v  = code_[n + i];
      auto    lb = lower_bound_[i];
      auto    s  = step_size_[i];

      float decode = lb + s * (v >> 4 & 0xF);
      float diff   = data[i] - decode;
      dis += diff * diff;
    }
    return dis;
  }

#endif

  // size_t size() const { return size; }

  ~SQ8() {
    delete[] lower_bound_;
    delete[] step_size_;
    delete[] code_;
  }

private:
  // size_t size_        = 0;
  size_t dim_         = 0;
  float *lower_bound_ = nullptr;
  float *step_size_   = nullptr;
  // uint8_t *cid_         = nullptr;
  uint8_t *code_ = nullptr;

  static inline Clusterer       clusterer;
  static inline MinMaxOptimizer optimizer;

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
};

}  // namespace posq