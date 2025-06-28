#pragma once

#include <omp.h>

#include "bitmap.h"
#include "clusterer.h"
#include "optimizer.h"

namespace pouq {

class Quantizer {
public:
  virtual ~Quantizer() = default;

  virtual float operator[](uint64_t i) const = 0;

  virtual uint64_t size() const = 0;
};

template <typename Clusterer = KrangeClusterer, typename Optimizer = PSOptimizer>
class POUQuantizer : public Quantizer {
public:
  explicit POUQuantizer(const float *data,
      const uint64_t                 size,
      const uint64_t                 c_bit,
      const uint64_t                 q_bit,
      const uint64_t                 groups    = 1,
      const bool                     opt_bound = true)
      : c_bit_(c_bit), q_bit_(q_bit), size_(size), groups_(groups) {
    this->step_size_   = new float[this->groups_ * (1 << this->c_bit_)];
    this->lower_bound_ = new float[this->groups_ * (1 << this->c_bit_)];
    this->cid_         = new uint8_t[(this->c_bit_ * this->size_ + 7) / 8];
    this->code_        = new uint8_t[(this->q_bit_ * this->size_ + 7) / 8];

    const auto div = static_cast<float>((1 << this->q_bit_) - 1);
#pragma omp parallel for
    for (uint64_t group = 0; group < this->groups_; group++) {
      const auto data_freq_map = this->count_freq(data, group);
      const auto bounds        = Clusterer()(1 << this->c_bit_, data_freq_map);
      const auto offset        = group * (1 << this->c_bit_);

      for (uint64_t i = 0; i < bounds.size(); i++) {
        auto [lower, upper] = bounds[i];
        if (opt_bound && lower < upper) {
          auto data_start = std::lower_bound(data_freq_map.begin(),
              data_freq_map.end(),
              lower,
              [](const std::pair<float, uint64_t> &lhs, const float rhs) -> bool { return lhs.first < rhs; });
          auto data_end   = std::upper_bound(data_freq_map.begin(),
              data_freq_map.end(),
              upper,
              [](const float rhs, const std::pair<float, uint64_t> &lhs) -> bool { return rhs < lhs.first; });

          const auto [opt_lower, opt_upper] = Optimizer()(div, lower, upper, data_start, data_end);
          lower                             = opt_lower;
          upper                             = opt_upper;
        }
        this->lower_bound_[offset + i] = lower;
        if (lower == upper) {
          this->step_size_[offset + i] = 1.0;
        } else {
          this->step_size_[offset + i] = (upper - lower) / div;
        }
      }

      static_cast<std::vector<std::pair<float, uint64_t>>>(data_freq_map).clear();
      for (uint64_t i = group; i < this->size_; i += this->groups_) {
        const float d  = data[i];
        const auto  it = std::upper_bound(
            bounds.begin(), bounds.end(), d, [](const float rhs, const std::pair<float, float> &lhs) -> bool {
              return rhs < lhs.first;
            });
        const uint64_t c = it - bounds.begin() - 1;
        bitmap::set(this->cid_, i, c, this->c_bit_);
        const float x =
            std::clamp((d - this->lower_bound_[offset + c]) / this->step_size_[offset + c] + 0.5f, 0.0f, div);
        bitmap::set(this->code_, i, static_cast<uint64_t>(x), this->q_bit_);
      }
    }
  }

  float operator[](uint64_t i) const override {
    const uint64_t group  = i % this->groups_;
    const uint64_t offset = bitmap::get(this->cid_, i, this->c_bit_) + group * (1 << this->c_bit_);
    const uint64_t x      = bitmap::get(this->code_, i, this->q_bit_);
    return this->lower_bound_[offset] + this->step_size_[offset] * static_cast<float>(x);
  }

  uint64_t size() const override { return this->size_; }

  ~POUQuantizer() override {
    delete[] this->lower_bound_;
    delete[] this->step_size_;
    delete[] this->cid_;
    delete[] this->code_;
  }

private:
  uint64_t c_bit_;
  uint64_t q_bit_;
  uint64_t size_;
  uint64_t groups_;
  float   *lower_bound_ = nullptr;
  float   *step_size_   = nullptr;
  uint8_t *cid_         = nullptr;
  uint8_t *code_        = nullptr;

  std::vector<std::pair<float, uint64_t>> count_freq(const float *data, const uint64_t group) const {
    std::vector<float> sorted_data;
    sorted_data.reserve(this->size_ / this->groups_);
    for (uint64_t i = group; i < this->size_; i += this->groups_) {
      sorted_data.push_back(data[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    float    current_value = sorted_data[0];
    uint64_t count         = 1;

    std::vector<std::pair<float, uint64_t>> data_freq_map;
    data_freq_map.reserve(sorted_data.size());
    for (uint64_t i = 1; i < sorted_data.size(); i++) {
      if (sorted_data[i] == current_value) {
        count++;
      } else {
        data_freq_map.emplace_back(current_value, count);
        current_value = sorted_data[i];
        count         = 1;
      }
    }

    data_freq_map.emplace_back(current_value, count);
    return data_freq_map;
  }
};

}  // namespace pouq