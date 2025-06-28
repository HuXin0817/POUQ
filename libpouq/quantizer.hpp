#pragma once

#include <omp.h>

#include "clusterer.hpp"
#include "optimizer.hpp"

#include <cstring>

namespace pouq {

class Quantizer {
public:
  virtual ~Quantizer() = default;

  virtual void train(const float *data, size_t size) = 0;

  virtual float operator[](size_t i) const = 0;

  virtual size_t size() const = 0;
};

class Float32Quantizer final : public Quantizer {
public:
  Float32Quantizer() = default;

  void train(const float *data, size_t size) override {
    size_ = size;
    data_ = new float[size_];
    std::memcpy(data_, data, size_ * sizeof(float));
  }

  float operator[](size_t i) const override { return data_[i]; }

  size_t size() const override { return size_; }

  ~Float32Quantizer() override { delete[] data_; }

private:
  size_t size_ = 0;
  float *data_ = nullptr;
};

template <typename Clusterer, typename Optimizer>
class QuantizerImpl : public Quantizer {
public:
  explicit QuantizerImpl(size_t c_bit, size_t q_bit, size_t groups) : c_bit_(c_bit), q_bit_(q_bit), groups_(groups) {}

  void train(const float *data, size_t size) override {
    size_          = size;
    step_size_     = new float[groups_ * (1 << c_bit_)];
    lower_bound_   = new float[groups_ * (1 << c_bit_)];
    cid_           = new uint8_t[size_];
    code_          = new uint8_t[size_];
    const auto div = static_cast<float>((1 << q_bit_) - 1);

#pragma omp parallel for default(none) shared(data, div)
    for (size_t group = 0; group < groups_; group++) {
      const auto data_freq_map = count_freq(data, group);
      const auto bounds        = clusterer(1 << c_bit_, data_freq_map);
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

          const auto [opt_lower, opt_upper] = optimizer(div, data_start, data_end);
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
      for (size_t i = group; i < size_; i += groups_) {
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

  float operator[](size_t i) const override {
    const size_t group  = i % groups_;
    const size_t offset = cid_[i] + group * (1 << c_bit_);
    const size_t x      = code_[i];
    return lower_bound_[offset] + step_size_[offset] * static_cast<float>(x);
  }

  size_t size() const override { return size_; }

  ~QuantizerImpl() override {
    delete[] lower_bound_;
    delete[] step_size_;
    delete[] cid_;
    delete[] code_;
  }

private:
  size_t   c_bit_       = 0;
  size_t   q_bit_       = 0;
  size_t   size_        = 0;
  size_t   groups_      = 0;
  float   *lower_bound_ = nullptr;
  float   *step_size_   = nullptr;
  uint8_t *cid_         = nullptr;
  uint8_t *code_        = nullptr;

  static inline Clusterer clusterer;
  static inline Optimizer optimizer;

  std::vector<std::pair<float, size_t>> count_freq(const float *data, const size_t group) const {
    std::vector<float> sorted_data;
    sorted_data.reserve(size_ / groups_);
    for (size_t i = group; i < size_; i += groups_) {
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

class SQQuantizer final : public QuantizerImpl<Clusterer, Optimizer> {
public:
  explicit SQQuantizer(size_t q_bit, size_t groups = 1) : QuantizerImpl(0, q_bit, groups) {}
};

class OSQQuantizer final : public QuantizerImpl<Clusterer, PSOptimizer> {
public:
  explicit OSQQuantizer(size_t q_bit, size_t groups = 1) : QuantizerImpl(0, q_bit, groups) {}
};

class LloydMaxQuantizer final : public QuantizerImpl<KmeansClusterer, CenterCalculator> {
public:
  explicit LloydMaxQuantizer(size_t c_bit, size_t groups = 1) : QuantizerImpl(c_bit, 0, groups) {}
};

class POUQQuantizer final : public QuantizerImpl<KrangeClusterer, PSOptimizer> {
public:
  explicit POUQQuantizer(size_t c_bit, size_t q_bit, size_t groups = 1) : QuantizerImpl(c_bit, q_bit, groups) {}
};

}  // namespace pouq