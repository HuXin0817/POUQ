#pragma once

#include <omp.h>

#include "optimizer.hpp"
#include "quantizer.hpp"
#include "segmenter.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <tuple>
#include <vector>

class Quantizer {
public:
  Quantizer() = default;

  explicit Quantizer(size_t dim) : dim_(dim) { assert(dim % 8 == 0); }

  void set_dim(size_t dim) {
    assert(dim % 8 == 0);
    dim_ = dim;
  }

  size_t dim() const { return dim_; }

  virtual ~Quantizer() = default;

  virtual void train(const float *data, size_t data_size) = 0;

  virtual float l2distance(const float *data, size_t data_index) const = 0;

protected:
  size_t dim_ = 0;
};

class Float32Quantizer final : public Quantizer {
public:
  explicit Float32Quantizer(size_t dim) : Quantizer(dim) {}

  void train(const float *data, size_t data_size) override {
    encode = new float[data_size];
    memcpy(encode, data, data_size * sizeof(float));
  }

  float l2distance(const float *data, size_t data_index) const override {
    // return ::l2distance(data, encode + data_index, dim_);
  }

  ~Float32Quantizer() override { delete[] encode; }

private:
  float *encode = nullptr;
};

template <typename Segmenter, typename Optimizer>
class QuantizerImpl : public Quantizer {
public:
  explicit QuantizerImpl(size_t c_bit, size_t q_bit, size_t dim) : Quantizer(dim), c_bit_(c_bit), q_bit_(q_bit) {}

  void train(const float *data, size_t size) override {
    step_size_     = new float[dim_ * (1 << c_bit_)];
    lower_bound_   = new float[dim_ * (1 << c_bit_)];
    cid_           = new uint8_t[size];
    code_          = new uint8_t[size];
    const auto div = static_cast<float>((1 << q_bit_) - 1);

#pragma omp parallel for
    for (size_t group = 0; group < dim_; group++) {
      const auto data_freq_map = count_freq(data, size, group);
      const auto bounds        = Segmenter()(1 << c_bit_, data_freq_map);
      const auto offset        = group * (1 << c_bit_);

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

          const auto [opt_lower, opt_upper] = Optimizer()(div, lower, upper, data_start, data_end);
          lower                             = opt_lower;
          upper                             = opt_upper;
        }
        lower_bound_[offset + i] = lower;
        if (lower == upper || div == 0.0f) {
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
        cid_[i]        = c;
        code_[i]       = std::clamp((d - lower_bound_[offset + c]) / step_size_[offset + c] + 0.5f, 0.0f, div);
      }
    }
  }

  float l2distance(const float *data, size_t data_index) const override {
    float dis = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
      auto dif = data[i] - operator[](i + data_index);
      dis += dif * dif;
    }
    return dis;
  }

  ~QuantizerImpl() override {
    delete[] lower_bound_;
    delete[] step_size_;
    delete[] cid_;
    delete[] code_;
  }

private:
  float operator[](size_t i) const {
    const size_t group  = i % dim_;
    const size_t offset = cid_[i] + group * (1 << c_bit_);
    const size_t x      = code_[i];
    return lower_bound_[offset] + step_size_[offset] * static_cast<float>(x);
  }

  size_t   c_bit_       = 0;
  size_t   q_bit_       = 0;
  float   *lower_bound_ = nullptr;
  float   *step_size_   = nullptr;
  uint8_t *cid_         = nullptr;
  uint8_t *code_        = nullptr;

  std::vector<std::pair<float, size_t>> count_freq(const float *data, const size_t size, const size_t group) const {
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

template <size_t nbit, typename Optimizer>
class UQQuantizer final : public QuantizerImpl<BlankSegmenter, Optimizer> {
public:
  explicit UQQuantizer(size_t dim) : QuantizerImpl<BlankSegmenter, Optimizer>(0, nbit, dim) {}
};

template <size_t nbit, typename Segmenter, typename Optimizer>
class POUQQuantizer final : public QuantizerImpl<Segmenter, Optimizer> {
public:
  explicit POUQQuantizer(size_t dim) : QuantizerImpl<Segmenter, Optimizer>(nbit / 2, nbit / 2, dim) {}
};

template <size_t nbit>
class LGBQuantizer final : public QuantizerImpl<KmeansSegmenter, CenterCalculator> {
public:
  explicit LGBQuantizer(size_t dim) : QuantizerImpl<KmeansSegmenter, CenterCalculator>(nbit, 0, dim) {}
};

class UQ4bitSIMDQuantizer final : public Quantizer {
public:
  explicit UQ4bitSIMDQuantizer(size_t dim) : Quantizer(dim) {
    lowers_     = static_cast<__m256 *>(_mm_malloc(dim_ * sizeof(__m256) / 8, 32));
    step_sizes_ = static_cast<__m256 *>(_mm_malloc(dim_ * sizeof(__m256) / 8, 32));
  }

  ~UQ4bitSIMDQuantizer() override {
    delete[] code;
    _mm_free(lowers_);
    _mm_free(step_sizes_);
  }

  void train(const float *data, size_t size) override {
    size_t n_samples = size / dim_;

    std::vector<float> lowers(dim_);
    std::vector<float> step_sizes(dim_);

    for (size_t d = 0; d < dim_; d++) {
      float min_val = data[d];
      float max_val = data[d];
      for (size_t i = 1; i < n_samples; i++) {
        float val = data[i * dim_ + d];
        min_val   = std::min(min_val, val);
        max_val   = std::max(max_val, val);
      }
      lowers[d]     = min_val;
      step_sizes[d] = (min_val == max_val) ? 1.0f : (max_val - min_val) / 15.0f;
    }

    for (size_t i = 0; i < dim_ / 8; i++) {
      float lower_array[8];
      float step_array[8];
      for (size_t j = 0; j < 8; j++) {
        size_t idx     = i * 8 + j;
        lower_array[j] = lowers[idx];
        step_array[j]  = step_sizes[idx];
      }
      lowers_[i]     = _mm256_loadu_ps(lower_array);
      step_sizes_[i] = _mm256_loadu_ps(step_array);
    }

    code = new uint32_t[size / 8];
    for (size_t i = 0; i < size; i += 8) {
      uint32_t packed = 0;
      for (int j = 0; j < 8 && i + j < size; j++) {
        size_t   idx = i + j;
        size_t   d   = idx % dim_;
        float    val = data[idx];
        uint32_t q   = std::min<uint32_t>(15, std::max<uint32_t>(0, std::round((val - lowers[d]) / step_sizes[d])));
        packed |= (q << ((7 - j) * 4));
      }
      code[i / 8] = packed;
    }
  }

  float l2distance(const float *data, size_t offset) const override {
    offset /= 8;
    __m256 sum_vec = _mm256_setzero_ps();

    for (size_t i = 0; i < dim_ / 8; i++) {
      const uint32_t       packed        = code[offset + i];
      const __m256i        packed_vec    = _mm256_set1_epi32(packed);
      static const __m256i shift_amounts = _mm256_setr_epi32(28, 24, 20, 16, 12, 8, 4, 0);
      static const __m256i mask          = _mm256_set1_epi32(0xF);
      __m256i              shifted       = _mm256_srlv_epi32(packed_vec, shift_amounts);
      __m256i              q_vals_vec    = _mm256_and_si256(shifted, mask);
      __m256               q_float       = _mm256_cvtepi32_ps(q_vals_vec);

      __m256 lower_vec     = lowers_[i];
      __m256 step_size_vec = step_sizes_[i];
      __m256 data_vec      = _mm256_loadu_ps(&data[i * 8]);

      __m256 decoded = _mm256_fmadd_ps(step_size_vec, q_float, lower_vec);
      __m256 diff    = _mm256_sub_ps(decoded, data_vec);
      sum_vec        = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    __m128 low128  = _mm256_castps256_ps128(sum_vec);
    __m128 high128 = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128  = _mm_add_ps(low128, high128);
    __m128 shuf    = _mm_movehdup_ps(sum128);
    sum128         = _mm_add_ps(sum128, shuf);
    shuf           = _mm_movehl_ps(shuf, sum128);
    sum128         = _mm_add_ss(sum128, shuf);
    return _mm_cvtss_f32(sum128);
  }

private:
  __m256   *lowers_     = nullptr;
  __m256   *step_sizes_ = nullptr;
  uint32_t *code        = nullptr;
};

class POUQ4bitSIMDQuantizer final : public Quantizer {

  struct ReconstructParameter {
    __m128 lower_bound;
    __m128 step_size;
  };

public:
  explicit POUQ4bitSIMDQuantizer(size_t dim) : Quantizer(dim) {
    bounds_data_   = nullptr;
    combined_data_ = nullptr;
  }

  void train(const float *data, size_t size) override {
    if (combined_data_) {
      _mm_free(combined_data_);
      combined_data_ = nullptr;
    }
    if (bounds_data_) {
      _mm_free(bounds_data_);
      bounds_data_ = nullptr;
    }

    size_t combined_data_size = size / 4;
    combined_data_            = static_cast<std::tuple<uint8_t, uint8_t, uint16_t> *>(
        _mm_malloc(combined_data_size * sizeof(std::tuple<uint8_t, uint8_t, uint16_t>), 256));
    if (!combined_data_) {
      throw std::bad_alloc();
    }

    size_t bounds_data_size = dim_ * 64;
    bounds_data_ =
        static_cast<ReconstructParameter *>(_mm_malloc(bounds_data_size * sizeof(ReconstructParameter), 256));
    if (!bounds_data_) {
      _mm_free(combined_data_);
      combined_data_ = nullptr;
      throw std::bad_alloc();
    }

    std::vector<float>    step_size(dim_ * 4);
    std::vector<float>    lower_bound(dim_ * 4);
    std::vector<uint8_t>  cid(size / 4);
    std::vector<uint16_t> code(size / 8);

    const size_t dim_div_4 = dim_ / 4;

#pragma omp parallel for
    for (size_t d = 0; d < dim_; d++) {
      const auto   data_freq_map = count_freq(data, size, d);
      const auto   bounds        = POUQSegmenter()(4, data_freq_map);
      const size_t d_times_4     = d * 4;

      for (size_t i = 0; i < bounds.size(); i++) {
        auto [lower, upper] = bounds[i];
        if (lower < upper) {
          const auto data_start = std::lower_bound(data_freq_map.begin(),
              data_freq_map.end(),
              lower,
              [](const std::pair<float, size_t> &lhs, const float rhs) -> bool { return lhs.first < rhs; });
          const auto data_end   = std::upper_bound(data_freq_map.begin(),
              data_freq_map.end(),
              upper,
              [](const float rhs, const std::pair<float, size_t> &lhs) -> bool { return rhs < lhs.first; });

          const auto [opt_lower, opt_upper] = PSOOptimizer()(3, lower, upper, data_start, data_end);
          lower                             = opt_lower;
          upper                             = opt_upper;
        }
        lower_bound[d_times_4 + i] = lower;
        if (lower == upper) {
          step_size[d_times_4 + i] = 1.0;
        } else {
          step_size[d_times_4 + i] = (upper - lower) / 3.0f;
        }
      }

      for (size_t i = d; i < size; i += dim_) {
        const auto it = std::upper_bound(
            bounds.begin(), bounds.end(), data[i], [](float rhs, const std::pair<float, float> &lhs) -> bool {
              return rhs < lhs.first;
            });
        const size_t c = it - bounds.begin() - 1;
        const float  x =
            std::clamp((data[i] - lower_bound[d_times_4 + c]) / step_size[d_times_4 + c] + 0.5f, 0.0f, 3.0f);
        const size_t base_index = (i / dim_) * dim_div_4;
        set(&cid[base_index], i % dim_, c);
        set16(&code[base_index / 2], i % dim_, x);
      }
    }

    for (size_t i = 0; i < size / 4; i += 2) {
      combined_data_[i / 2] = std::make_tuple(cid[i], cid[i + 1], code[i / 2]);
    }

#pragma omp parallel for
    for (size_t g = 0; g < dim_ / 4; g++) {
      for (size_t j = 0; j < 256; j++) {
        const auto [x0, x1, x2, x3] = get(j);
        const size_t base_idx       = g * 16;
        const __m128 lb             = _mm_setr_ps(lower_bound[base_idx + 0 * 4 + x0],
            lower_bound[base_idx + 1 * 4 + x1],
            lower_bound[base_idx + 2 * 4 + x2],
            lower_bound[base_idx + 3 * 4 + x3]);
        const __m128 st             = _mm_setr_ps(step_size[base_idx + 0 * 4 + x0],
            step_size[base_idx + 1 * 4 + x1],
            step_size[base_idx + 2 * 4 + x2],
            step_size[base_idx + 3 * 4 + x3]);
        bounds_data_[g * 256 + j]   = {lb, st};
      }
    }
  }

  float l2distance(const float *data, size_t offset) const override {
    offset /= 4;
    __m256 sum_vec = _mm256_setzero_ps();

    static const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    static const __m256i mask   = _mm256_set1_epi32(3);

    for (size_t i = 0; i < dim_; i += 8) {
      const size_t idx            = i / 4;
      const auto [c1, c2, code]   = combined_data_[(offset + idx) / 2];
      const auto [lb1, st1]       = bounds_data_[idx * 256 + c1];
      const auto [lb2, st2]       = bounds_data_[(idx + 1) * 256 + c2];
      const __m256  lb_vec        = _mm256_insertf128_ps(_mm256_castps128_ps256(lb1), lb2, 1);
      const __m256  st_vec        = _mm256_insertf128_ps(_mm256_castps128_ps256(st1), st2, 1);
      const __m256i bytes         = _mm256_set1_epi32(code);
      const __m256i shifted       = _mm256_srlv_epi32(bytes, shifts);
      const __m256i masked        = _mm256_and_si256(shifted, mask);
      const __m256  code_vec      = _mm256_cvtepi32_ps(masked);
      const __m256  reconstructed = _mm256_fmadd_ps(code_vec, st_vec, lb_vec);
      const __m256  data_vec      = _mm256_loadu_ps(data + i);
      const __m256  diff          = _mm256_sub_ps(reconstructed, data_vec);
      sum_vec                     = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    __m128 low128  = _mm256_castps256_ps128(sum_vec);
    __m128 high128 = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128  = _mm_add_ps(low128, high128);
    __m128 shuf    = _mm_movehdup_ps(sum128);
    sum128         = _mm_add_ps(sum128, shuf);
    shuf           = _mm_movehl_ps(shuf, sum128);
    sum128         = _mm_add_ss(sum128, shuf);
    return _mm_cvtss_f32(sum128);
  }

  ~POUQ4bitSIMDQuantizer() override {
    if (combined_data_) {
      _mm_free(combined_data_);
    }
    if (bounds_data_) {
      _mm_free(bounds_data_);
    }
  }

private:
  ReconstructParameter                   *bounds_data_   = nullptr;
  std::tuple<uint8_t, uint8_t, uint16_t> *combined_data_ = nullptr;

  std::vector<std::pair<float, size_t>> count_freq(const float *data, size_t size, size_t group) const {
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

  static void set(uint8_t *data, size_t i, size_t n) {
    const size_t offset = (i & 3) << 1;
    i >>= 2;
    data[i] &= ~(3 << offset);
    data[i] |= n << offset;
  }

  static std::tuple<size_t, size_t, size_t, size_t> get(uint8_t byte) {
    return {
        byte & 3,
        byte >> 2 & 3,
        byte >> 4 & 3,
        byte >> 6 & 3,
    };
  }

  static void set16(uint16_t *data, size_t i, size_t n) {
    const size_t offset = (i & 7) << 1;
    i >>= 3;
    data[i] &= ~(3 << offset);
    data[i] |= n << offset;
  }

  static std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t> get16(uint16_t value) {
    return {
        value & 3,
        (value >> 2) & 3,
        (value >> 4) & 3,
        (value >> 6) & 3,
        (value >> 8) & 3,
        (value >> 10) & 3,
        (value >> 12) & 3,
        (value >> 14) & 3,
    };
  }
};