#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

class ScalarQuantizer {
  static constexpr uint32_t kLevel = 15.0f;

 public:
  ScalarQuantizer() = default;

  ~ScalarQuantizer() = default;

  void Train(const std::vector<std::vector<float>>& data);

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
  uint32_t n_sample_ = 0;
  uint32_t n_dim_ = 0;

  std::vector<float> lower_;
  std::vector<float> step_size_;
  std::vector<std::vector<uint8_t>> code_;
};
