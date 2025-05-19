#pragma once

#include <omp.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>

#include "bitmap.h"
#include "krange.h"
#include "pso.h"

namespace py = pybind11;

std::vector<std::pair<float, uint64_t>> count_freq(const py::array_t<float> &data) {
  std::vector sorted_data(data.data(), data.data() + data.size());
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

class QVector {
public:
  explicit QVector(const py::array_t<float> &data,
      const uint64_t                         c_bit,
      const uint64_t                         q_bit,
      const bool                             learn_step_size   = true,
      const uint64_t                         max_iterations    = 128,
      const uint64_t                         grid_side_length  = 8,
      const float                            grid_scale_factor = 0.1f,
      const float                            initial_inertia   = 0.9f,
      const float                            final_inertia     = 0.4f,
      const float                            c1                = 1.8f,
      const float                            c2                = 1.8f) {
    c_bit_        = c_bit;
    q_bit_        = q_bit;
    ndim_         = data.ndim();
    ssize_t size_ = data.size();
    step_size_    = new float[1 << c_bit_];
    lower_bound_  = new float[1 << c_bit_];
    cid_          = new uint8_t[(c_bit_ * size_ + 7) / 8];
    code_         = new uint8_t[(q_bit_ * size_ + 7) / 8];
    shape_        = new uint64_t[ndim_];
    for (ssize_t i = 0; i < ndim_; i++) {
      shape_[i] = data.shape(i);
    }

    const auto  div           = static_cast<float>((1 << q_bit_) - 1);
    const auto  data_freq_map = count_freq(data);
    const auto  bounds        = krange(1 << c_bit_, data_freq_map);
    const auto &lowers        = bounds.first;
    const auto &uppers        = bounds.second;

#pragma omp parallel for
    for (uint64_t c = 0; c < lowers.size(); c++) {
      float lower = lowers[c];
      float upper = uppers[c];
      if (learn_step_size && lower < upper) {
        auto begin = std::lower_bound(data_freq_map.begin(),
            data_freq_map.end(),
            lower,
            [](const std::pair<float, uint64_t> &lhs, const float rhs) { return lhs.first < rhs; });
        auto end   = std::upper_bound(data_freq_map.begin(),
            data_freq_map.end(),
            upper,
            [](const float rhs, const std::pair<float, uint64_t> &lhs) { return rhs < lhs.first; });

        const auto [opt_lower, opt_upper] = pso_optimize(div,
            lower,
            upper,
            begin,
            end,
            max_iterations,
            grid_side_length,
            grid_scale_factor,
            initial_inertia,
            final_inertia,
            c1,
            c2);

        lower = opt_lower;
        upper = opt_upper;
      }
      lower_bound_[c] = lower;
      if (lower == upper) {
        step_size_[c] = 1.0;
      } else {
        step_size_[c] = (upper - lower) / div;
      }
    }

#pragma omp parallel for
    for (uint64_t i = 0; i < size_; i++) {
      const float    d  = data.data()[i];
      const auto     it = std::upper_bound(lowers.begin(), lowers.end(), d);
      const uint64_t c  = it - lowers.begin() - 1;
      bitmap::set(cid_, i, c, c_bit_);
      const float x = std::clamp((d - lower_bound_[c]) / step_size_[c] + 0.5f, 0.0f, div);
      bitmap::set(code_, i, x, q_bit_);
    }
  }

  float at(uint64_t i) const {
    const uint64_t c = bitmap::get(cid_, i, c_bit_);
    const uint64_t x = bitmap::get(code_, i, q_bit_);
    return lower_bound_[c] + step_size_[c] * static_cast<float>(x);
  }

  uint64_t ndim() const { return ndim_; }

  uint64_t shape(uint64_t i) const { return shape_[i]; }

  ~QVector() {
    delete[] lower_bound_;
    delete[] step_size_;
    delete[] code_;
    delete[] cid_;
  }

private:
  uint64_t  c_bit_;
  uint64_t  q_bit_;
  uint64_t  ndim_;
  uint64_t *shape_;

  float *lower_bound_ = nullptr;
  float *step_size_   = nullptr;

  uint8_t *cid_  = nullptr;
  uint8_t *code_ = nullptr;
};