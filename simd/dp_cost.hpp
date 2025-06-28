#pragma once

#include <algorithm>
#include <limits>
#include <vector>

// SIMD headers
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define SIMD_ARM_NEON
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_AVX2
#endif

namespace pouq::simd {

#ifdef SIMD_ARM_NEON
// ARM NEON optimized version
inline std::pair<float, size_t> dp_cost_simd_neon(size_t j,
    size_t                                               mid,
    size_t                                               opt_l,
    size_t                                               opt_r,
    const std::pair<float, size_t>                      *data,
    const float                                         *cnt,
    const float                                         *dp) {
  const size_t start     = std::max(j - 1, opt_l);
  const size_t end       = std::min(mid - 1, opt_r);
  float        min_cost  = std::numeric_limits<float>::max();
  size_t       split_pos = 0;

  const float data_mid = static_cast<float>(data[mid - 1].first);
  const float cnt_mid  = cnt[mid];

  // NEON SIMD processing for 4 elements at a time
  uint32_t     m        = start;
  const size_t simd_end = start + ((end - start + 1) / 4) * 4;

  if (simd_end > start) {
    float32x4_t min_costs = vdupq_n_f32(std::numeric_limits<float>::max());
    uint32x4_t  indices   = {0, 0, 0, 0};

    for (; m < simd_end; m += 4) {
      // Load data values
      float32x4_t data_vals = {static_cast<float>(data[m].first),
          static_cast<float>(data[m + 1].first),
          static_cast<float>(data[m + 2].first),
          static_cast<float>(data[m + 3].first)};

      // Load cnt and dp values
      float32x4_t cnt_vals = {cnt[m], cnt[m + 1], cnt[m + 2], cnt[m + 3]};
      float32x4_t dp_vals  = {dp[m], dp[m + 1], dp[m + 2], dp[m + 3]};

      // Calculate w = data_mid - data_vals
      float32x4_t w = vsubq_f32(vdupq_n_f32(data_mid), data_vals);

      // Calculate c = cnt_mid - cnt_vals
      float32x4_t c = vsubq_f32(vdupq_n_f32(cnt_mid), cnt_vals);

      // Calculate cost = dp + w * w * c
      float32x4_t w_squared = vmulq_f32(w, w);
      float32x4_t cost      = vfmaq_f32(dp_vals, w_squared, c);

      // Find minimum and update indices
      uint32x4_t mask            = vcltq_f32(cost, min_costs);
      min_costs                  = vbslq_f32(mask, cost, min_costs);
      uint32x4_t current_indices = {m, m + 1, m + 2, m + 3};
      indices                    = vbslq_u32(mask, current_indices, indices);
    }

    // Extract minimum from SIMD register
    float    costs[4];
    uint32_t idx[4];
    vst1q_f32(costs, min_costs);
    vst1q_u32(idx, indices);

    for (int i = 0; i < 4; ++i) {
      if (costs[i] < min_cost) {
        min_cost  = costs[i];
        split_pos = idx[i];
      }
    }
  }

  // Process remaining elements
  for (; m <= end; ++m) {
    const float w    = data_mid - static_cast<float>(data[m].first);
    const float c    = cnt_mid - cnt[m];
    const float cost = dp[m] + w * w * c;
    if (cost < min_cost) {
      min_cost  = cost;
      split_pos = m;
    }
  }

  return {min_cost, split_pos};
}
#endif

#ifdef SIMD_AVX2
// AVX2 optimized version
inline std::pair<float, size_t> dp_cost_simd_avx2(size_t j,
    size_t                                               mid,
    size_t                                               opt_l,
    size_t                                               opt_r,
    const std::pair<float, size_t>                      *data,
    const float                                         *cnt,
    const float                                         *dp) {
  const size_t start     = std::max(j - 1, opt_l);
  const size_t end       = std::min(mid - 1, opt_r);
  float        min_cost  = std::numeric_limits<float>::max();
  size_t       split_pos = 0;

  const float data_mid = static_cast<float>(data[mid - 1].first);
  const float cnt_mid  = cnt[mid];

  // AVX2 SIMD processing for 8 elements at a time
  size_t       m        = start;
  const size_t simd_end = start + ((end - start + 1) / 8) * 8;

  if (simd_end > start) {
    __m256  min_costs = _mm256_set1_ps(std::numeric_limits<float>::max());
    __m256i indices   = _mm256_setzero_si256();

    for (; m < simd_end; m += 8) {
      // Load data values
      __m256 data_vals = _mm256_set_ps(static_cast<float>(data[m + 7].first),
          static_cast<float>(data[m + 6].first),
          static_cast<float>(data[m + 5].first),
          static_cast<float>(data[m + 4].first),
          static_cast<float>(data[m + 3].first),
          static_cast<float>(data[m + 2].first),
          static_cast<float>(data[m + 1].first),
          static_cast<float>(data[m].first));

      // Load cnt and dp values
      __m256 cnt_vals = _mm256_loadu_ps(&cnt[m]);
      __m256 dp_vals  = _mm256_loadu_ps(&dp[m]);

      // Calculate w = data_mid - data_vals
      __m256 w = _mm256_sub_ps(_mm256_set1_ps(data_mid), data_vals);

      // Calculate c = cnt_mid - cnt_vals
      __m256 c = _mm256_sub_ps(_mm256_set1_ps(cnt_mid), cnt_vals);

      // Calculate cost = dp + w * w * c
      __m256 w_squared = _mm256_mul_ps(w, w);
      __m256 cost      = _mm256_fmadd_ps(w_squared, c, dp_vals);

      // Find minimum and update indices
      __m256 mask             = _mm256_cmp_ps(cost, min_costs, _CMP_LT_OQ);
      min_costs               = _mm256_blendv_ps(min_costs, cost, mask);
      __m256i current_indices = _mm256_set_epi32(m + 7, m + 6, m + 5, m + 4, m + 3, m + 2, m + 1, m);
      indices                 = _mm256_blendv_epi8(indices, current_indices, _mm256_castps_si256(mask));
    }

    // Extract minimum from SIMD register
    float costs[8];
    int   idx[8];
    _mm256_storeu_ps(costs, min_costs);
    _mm256_storeu_si256((__m256i *)idx, indices);

    for (int i = 0; i < 8; ++i) {
      if (costs[i] < min_cost) {
        min_cost  = costs[i];
        split_pos = idx[i];
      }
    }
  }

  // Process remaining elements
  for (; m <= end; ++m) {
    const float w    = data_mid - static_cast<float>(data[m].first);
    const float c    = cnt_mid - cnt[m];
    const float cost = dp[m] + w * w * c;
    if (cost < min_cost) {
      min_cost  = cost;
      split_pos = m;
    }
  }

  return {min_cost, split_pos};
}
#endif

// Generic fallback version
inline std::pair<float, size_t> dp_cost_simd_generic(size_t j,
    size_t                                                  mid,
    size_t                                                  opt_l,
    size_t                                                  opt_r,
    const std::pair<float, size_t>                         *data,
    const float                                            *cnt,
    const float                                            *dp) {
  const size_t start     = std::max(j - 1, opt_l);
  const size_t end       = std::min(mid - 1, opt_r);
  float        min_cost  = std::numeric_limits<float>::max();
  size_t       split_pos = 0;
  for (size_t m = start; m <= end; ++m) {
    const float w    = static_cast<float>(data[mid - 1].first) - static_cast<float>(data[m].first);
    const float c    = cnt[mid] - cnt[m];
    const float cost = dp[m] + w * w * c;
    if (cost < min_cost) {
      min_cost  = cost;
      split_pos = m;
    }
  }
  return {min_cost, split_pos};
}

// Main function that dispatches to the appropriate SIMD version
inline std::pair<float, size_t> dp_cost_simd(size_t j,
    size_t                                          mid,
    size_t                                          opt_l,
    size_t                                          opt_r,
    const std::pair<float, size_t>                 *data,
    const float                                    *cnt,
    const float                                    *dp) {
#ifdef SIMD_ARM_NEON
  return dp_cost_simd_neon(j, mid, opt_l, opt_r, data, cnt, dp);
#elif defined(SIMD_AVX2)
  return dp_cost_simd_avx2(j, mid, opt_l, opt_r, data, cnt, dp);
#else
  return dp_cost_simd_generic(j, mid, opt_l, opt_r, data, cnt, dp);
#endif
}

};  // namespace pouq::simd