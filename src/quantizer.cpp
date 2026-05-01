#include "quantizer.h"

#include <tbb/parallel_for.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ranges>
#include <vector>

#include "clusterer.h"

namespace pouq {

static uint8_t Packing(const uint8_t* x) { return (x[0] << 6) | (x[1] << 4) | (x[2] << 2) | x[3]; }

static std::array<uint8_t, 4> Unpack(uint8_t code) {
  return {
      static_cast<uint8_t>((code >> 6) & 0x03),
      static_cast<uint8_t>((code >> 4) & 0x03),
      static_cast<uint8_t>((code >> 2) & 0x03),
      static_cast<uint8_t>(code & 0x03),
  };
}

void Quantizer::Train(uint32_t n_sample, uint32_t n_dim, const float* data) {
  Clear();

  n_sample_ = n_sample;
  n_dim_ = n_dim;

  if (n_sample_ == 0) {
    return;
  }

  if (n_dim_ == 0) {
    return;
  }

  n_padding_dim_ = (n_dim_ + kAligned - 1) / kAligned * kAligned;

  std::vector<std::vector<uint8_t>> codebook(n_sample_, std::vector<uint8_t>(n_padding_dim_, 0));
  std::vector<std::vector<uint8_t>> reconstructed_param_index(n_sample_, std::vector<uint8_t>(n_padding_dim_, 0));
  std::vector<std::vector<float>> lower(n_padding_dim_, std::vector<float>(kClusterNumber, 0.0f));
  std::vector<std::vector<float>> step_size(n_padding_dim_, std::vector<float>(kClusterNumber, 1.0f));

  tbb::parallel_for<uint32_t>(0, n_dim_, [&](const uint32_t dim) {
    std::vector<float> dim_data(n_sample_, 0.0f);
    for (uint32_t i = 0; i < n_sample_; i++) {
      dim_data[i] = data[i * n_dim + dim];
    }
    std::ranges::sort(dim_data);

    auto [lefts, rights] = clusterer_.Split(dim_data);
    for (size_t i = 0; i < lefts.size(); i++) {
      float& l = lefts[i];
      float& r = rights[i];
      assert(l <= r);

      if (r - l > std::numeric_limits<float>::epsilon()) {
        const auto l_iter = std::ranges::lower_bound(dim_data, l);
        const auto r_iter = std::ranges::upper_bound(dim_data, r);
        std::span<float> range(l_iter, r_iter);
        assert(range.front() == l && range.back() == r);
        auto [opt_l, opt_r] = optimizer_.Optimize(range);
        l = opt_l;
        r = opt_r;
      }

      lower[dim][i] = l;
      if (r - l > std::numeric_limits<float>::epsilon()) {
        step_size[dim][i] = (r - l) / kLevel;
      } else {
        step_size[dim][i] = 1.0f;
      }
    }

    for (uint32_t i = 0; i < n_sample_; i++) {
      float d = data[i * n_dim + dim];

      uint8_t index = 0;
      float min_distance = std::numeric_limits<float>::max();
      for (uint32_t j = 0; j < kClusterNumber; j++) {
        float middle = lower[dim][j] + step_size[dim][j] * kLevel / 2.0f;
        float distance = std::fabs(d - middle);
        if (distance < min_distance) {
          min_distance = distance;
          index = j;
        }
      }

      reconstructed_param_index[i][dim] = index;
      float code = std::roundf((d - lower[dim][index]) / step_size[dim][index]);
      code = std::clamp(code, 0.0f, kLevel);
      codebook[i][dim] = static_cast<uint8_t>(code);
    }
  });

  lower_.resize(kPackage * n_padding_dim_ / kAligned);
  step_size_.resize(kPackage * n_padding_dim_ / kAligned);
  code_.resize(n_sample_, std::vector<Code>(n_padding_dim_ / kAligned, Code(0, 0)));

  tbb::parallel_for<uint32_t>(0, n_padding_dim_ / kAligned, [&](const uint32_t i) {
    uint32_t dim = i * kAligned;
    uint32_t offset = i * kPackage;
    for (uint32_t j = 0; j < kPackage; j++) {
      auto [x0, x1, x2, x3] = Unpack(j);
      lower_[offset + j] = {
          lower[dim + 0][x0],
          lower[dim + 1][x1],
          lower[dim + 2][x2],
          lower[dim + 3][x3],
      };
      step_size_[offset + j] = {
          step_size[dim + 0][x0],
          step_size[dim + 1][x1],
          step_size[dim + 2][x2],
          step_size[dim + 3][x3],
      };
    }

    for (uint32_t j = 0; j < n_sample_; j++) {
      uint8_t reconstructed_param_index_package = Packing(reconstructed_param_index[j].data() + dim);
      uint8_t code_package = Packing(codebook[j].data() + dim);
      code_[j][i] = {reconstructed_param_index_package, code_package};
    }
  });
}

void Quantizer::ForBatch(uint32_t n, const std::function<bool(uint32_t, const m128&)>& f) {
  for (uint32_t i = 0; i < n_padding_dim_ / kAligned; i++) {
    uint32_t offset = i * kPackage;
    auto [reconstructed_param_index_package, code_package] = code_[n][i];
    m128 lower = lower_[offset + reconstructed_param_index_package];
    m128 step_size = step_size_[offset + reconstructed_param_index_package];
    auto [c0, c1, c2, c3] = Unpack(code_package);
    m128 code = {
        static_cast<float>(c0),
        static_cast<float>(c1),
        static_cast<float>(c2),
        static_cast<float>(c3),
    };
    m128 decode = xsimd::fma(code, step_size, lower);
    if (f(i, decode)) {
      break;
    }
  }
}

void Quantizer::Decode(uint32_t n, float* data) {
  ForBatch(n, [&](uint32_t i, const m128& decode) -> bool {
    uint32_t dim = i * kAligned;
    if (dim + kAligned <= n_dim_) {
      decode.store_unaligned(data + dim);
    } else {
      for (uint32_t j = dim; j < n_dim_; j++) {
        data[j] = decode.get(j - dim);
      }
    }
    return false;
  });
}

float Quantizer::Distance(uint32_t n, const float* data) {
  float distance = 0.0f;
  ForBatch(n, [&](uint32_t i, const m128& decode) -> bool {
    uint32_t dim = i * kAligned;
    if (dim + kAligned <= n_dim_) {
      m128 data_batch = m128::load_unaligned(data + dim);
      m128 diff = data_batch - decode;
      distance += xsimd::reduce_add(diff * diff);
    } else {
      for (uint32_t j = dim; j < n_dim_; j++) {
        float d = decode.get(j - dim);
        float diff = data[j] - d;
        distance += diff * diff;
      }
    }
    return false;
  });
  return distance;
}

void Quantizer::Clear() {
  n_sample_ = 0;
  n_dim_ = 0;
  n_padding_dim_ = 0;

  lower_.clear();
  step_size_.clear();
  code_.clear();
}

}  // namespace pouq