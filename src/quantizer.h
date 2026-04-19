#pragma once

#include <functional>
#include <vector>
#include <xsimd/xsimd.hpp>

#include "clusterer.h"
#include "optimizer.h"
#include "util.h"

namespace pouq {

class Quantizer {
 public:
  static constexpr uint32_t kAligned = 4;
  static constexpr uint32_t kPackage = 256;  // std::pow(kAligned, kAligned);
  static constexpr float kLevel = 3.0f;

  static constexpr uint32_t kClusterNumber = 4;

  using Code = std::pair<uint8_t, uint8_t>;

  Quantizer() : clusterer_(kClusterNumber), optimizer_(kLevel) {}

  ~Quantizer() = default;

  void Train(const std::vector<std::vector<float>>& data);

  void ForBatch(uint32_t n, std::function<bool(uint32_t, const m128&)> f);

  void Decode(uint32_t n, float* data);

  void Decode(uint32_t n, std::vector<float>& data) {
    assert(data.size() >= n_dim_);
    Decode(n, data.data());
  }

  std::vector<float> Decode(uint32_t n) {
    std::vector<float> result(n_dim_);
    Decode(n, result);
    return result;
  }

  float Distance(uint32_t n, const float* data);

  float Distance(uint32_t n, const std::vector<float>& data) {
    assert(data.size() >= n_dim_);
    return Distance(n, data.data());
  }

  void Clear();

 private:
  cluster::Clusterer clusterer_;
  optimize::Optimizer optimizer_;

  uint32_t n_sample_ = 0;
  uint32_t n_dim_ = 0;
  uint32_t n_padding_dim_ = 0;

  std::vector<m128> lower_;
  std::vector<m128> step_size_;
  std::vector<std::vector<Code>> code_;
};

}  // namespace pouq