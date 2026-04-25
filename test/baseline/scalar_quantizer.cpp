#include "scalar_quantizer.h"

#include <tbb/parallel_for.h>

#include <xsimd/xsimd.hpp>

void ScalarQuantizer::Train(const std::vector<std::vector<float>>& data) {
  Clear();

  n_sample_ = data.size();
  assert(n_sample_ > 0);

  n_dim_ = data[0].size();
  assert(n_dim_ > 0);

  lower_.resize(n_dim_);
  step_size_.resize(n_dim_);
  code_.resize(n_sample_, std::vector<uint8_t>(n_dim_));

  tbb::parallel_for(static_cast<uint32_t>(0), n_dim_, [&](const uint32_t dim) {
    float l = data[0][dim];
    float r = data[0][dim];
    for (uint32_t j = 0; j < n_sample_; j++) {
      l = std::min(l, data[j][dim]);
      r = std::max(r, data[j][dim]);
    }

    lower_[dim] = l;
    if (r - l > std::numeric_limits<float>::epsilon()) {
      step_size_[dim] = (r - l) / kLevel;
    }

    for (uint32_t j = 0; j < n_sample_; j++) {
      code_[j][dim] = (data[j][dim] - lower_[dim]) / step_size_[dim] + 0.5f;
    }
  });
}

void ScalarQuantizer::Decode(uint32_t n, float* data) {
  for (uint32_t j = 0; j < n_dim_; j++) {
    float code = code_[n][j];
    float lower = lower_[j];
    float step_size = step_size_[j];
    float decode = code * step_size + lower;
    data[j] = decode;
  }
}

float ScalarQuantizer::Distance(uint32_t n, const float* data) {
  float distance = 0.0f;
  for (uint32_t j = 0; j < n_dim_; j++) {
    float d = data[j];
    float code = code_[n][j];
    float lower = lower_[j];
    float step_size = step_size_[j];
    float decode = code * step_size + lower;
    float diff = decode - d;
    distance += diff * diff;
  }

  return distance;
}

void ScalarQuantizer::Clear() {
  n_sample_ = 0;
  n_dim_ = 0;

  lower_.clear();
  step_size_.clear();
  code_.clear();
}