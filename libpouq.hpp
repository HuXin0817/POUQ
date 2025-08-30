#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#define POUQ_X86_64 1
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define POUQ_ARM64 1
#include <arm_neon.h>
#endif

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <memory>
#include <random>
#include <vector>

namespace pouq {

class Quantizer {
  static std::vector<std::pair<float, float>>
  segment(int k, const std::vector<std::pair<float, int>>& data_freq_map) {
    assert(k > 0);
    assert(!data_freq_map.empty());

    int size = static_cast<int>(data_freq_map.size());
    k = std::min(size, k);

    std::vector<int> sum_count(size + 1, 0);
    for (int i = 1; i <= size; ++i) {
      sum_count[i] = sum_count[i - 1] + data_freq_map[i - 1].second;
    }

    std::vector<float> prev_dp(size + 1, FLT_MAX);
    std::vector<float> curr_dp(size + 1, FLT_MAX);
    std::vector<std::vector<int>> prev_idx(size + 1, std::vector<int>(k + 1, 0));
    prev_dp[0] = 0.0f;

    struct Task {
      int j;
      int left;
      int right;
      int opt_left;
      int opt_right;
    };

    for (int j = 1; j <= k; ++j) {
      std::vector<Task> tasks{{j, j, size, 0, size - 1}};
      tasks.reserve(size);

      while (!tasks.empty()) {
        auto [j, l, r, opt_l, opt_r] = tasks.back();
        tasks.pop_back();
        if (l > r) {
          continue;
        }

        int mid = (l + r) / 2;
        int start = std::max(j - 1, opt_l);
        int end = std::min(mid - 1, opt_r);
        float min_cost = FLT_MAX;
        int split_pos = 0;
        for (int m = start; m <= end; ++m) {
          float width = static_cast<float>(data_freq_map[mid - 1].first) -
                        static_cast<float>(data_freq_map[m].first);
          int count = sum_count[mid] - sum_count[m];
          float cost = prev_dp[m] + width * width * static_cast<float>(count);
          if (cost < min_cost) {
            min_cost = cost;
            split_pos = m;
          }
        }

        curr_dp[mid] = min_cost;
        prev_idx[mid][j] = split_pos;
        if (l < r) {
          tasks.push_back({j, mid + 1, r, split_pos, opt_r});
          tasks.push_back({j, l, mid - 1, opt_l, split_pos});
        }
      }

      std::swap(prev_dp, curr_dp);
      std::fill(curr_dp.begin(), curr_dp.end(), FLT_MAX);
    }

    std::vector<int> split_pos(k);
    int curr_pos = size;
    for (int j = k; j > 0; --j) {
      int m = prev_idx[curr_pos][j];
      split_pos[j - 1] = m;
      curr_pos = m;
    }

    std::vector<std::pair<float, float>> bounds(k);
    for (int t = 0; t < k; ++t) {
      int start = split_pos[t];
      int end = t < k - 1 ? split_pos[t + 1] - 1 : size - 1;
      bounds[t] = {data_freq_map[start].first, data_freq_map[end].first};
    }

    return bounds;
  }

  static float
  loss(float div,
       float lower,
       float step,
       const std::vector<std::pair<float, int>>::iterator& data_begin,
       const std::vector<std::pair<float, int>>::iterator& data_end) {
    assert(div > 0.0f);
    assert(step >= FLT_EPSILON);
    assert(data_begin <= data_end);

    float total_loss = 0.0f;

    for (auto it = data_begin; it != data_end; ++it) {
      auto [data_value, point_count] = *it;
      float real_quantized_code = (data_value - lower) / step;
      float quantized_code = 0.0f;

      if (data_value > lower) {
        quantized_code = std::round(real_quantized_code);
        if (quantized_code > div) {
          quantized_code = div;
        }
      }

      float code_loss = real_quantized_code - quantized_code;
      total_loss += code_loss * code_loss * static_cast<float>(point_count);
    }

    return total_loss * step * step;
  }

  static std::pair<float, float>
  optimize(float div,
           float init_lower,
           float init_upper,
           const std::vector<std::pair<float, int>>::iterator& data_begin,
           const std::vector<std::pair<float, int>>::iterator& data_end,
           int max_iter,
           int particle_count,
           float scale_factor,
           float init_inertia,
           float final_inertia,
           float init_c1,
           float final_c1,
           float init_c2,
           float final_c2) {
    assert(div > 0.0f);
    assert(init_lower <= init_upper);
    assert(data_begin <= data_end);
    assert(max_iter >= 0);
    assert(particle_count >= 0);
    assert(scale_factor >= 0.0f);
    assert(init_inertia >= 0.0f);
    assert(final_inertia >= 0.0f);
    assert(init_c1 >= 0.0f);
    assert(final_c1 >= 0.0f);
    assert(init_c2 >= 0.0f);
    assert(final_c2 >= 0.0f);

    float init_range_width = init_upper - init_lower;
    float init_step = init_range_width / div;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution v_dis(-init_range_width * 0.1f, init_range_width * 0.1f);
    std::uniform_real_distribution p_dis(0.0f, 1.0f);
    std::uniform_real_distribution lower_dis(init_lower - init_range_width * scale_factor,
                                             init_lower + init_range_width * scale_factor);
    std::uniform_real_distribution step_dis(init_step * (1.0f - scale_factor),
                                            init_step * (1.0f + scale_factor));

    struct Particle {
      float lower;
      float step;
      float v_lower;
      float v_step;
      float best_lower;
      float best_step;
      float min_loss;
    };

    float global_best_lower = init_lower;
    float global_best_step = init_step;
    float global_min_loss = loss(div, init_lower, init_step, data_begin, data_end);

    std::vector<Particle> swarm(particle_count);
    for (auto& particle : swarm) {
      float lower = lower_dis(gen);
      float step = step_dis(gen);
      float v_lower = v_dis(gen);
      float v_step = v_dis(gen);
      float min_loss = loss(div, lower, step, data_begin, data_end);

      particle = {
          .lower = lower,
          .step = step,
          .v_lower = v_lower,
          .v_step = v_step,
          .best_lower = lower,
          .best_step = step,
          .min_loss = min_loss,
      };

      if (min_loss < global_min_loss) {
        global_min_loss = min_loss;
        global_best_lower = particle.lower;
        global_best_step = particle.step;
      }
    }

    for (int iter = 0; iter < max_iter; ++iter) {
      float x = static_cast<float>(iter) / static_cast<float>(max_iter);
      float inertia = init_inertia - (init_inertia - final_inertia) * x;
      float c1 = init_c1 - (init_c1 - final_c1) * x;
      float c2 = init_c2 - (init_c2 - final_c2) * x;

      for (auto& particle : swarm) {
        float r1 = p_dis(gen);
        float r2 = p_dis(gen);

        particle.v_lower = inertia * particle.v_lower +
                           c1 * r1 * (particle.best_lower - particle.lower) +
                           c2 * r2 * (global_best_lower - particle.lower);

        particle.v_step = inertia * particle.v_step +
                          c1 * r1 * (particle.best_step - particle.step) +
                          c2 * r2 * (global_best_step - particle.step);

        particle.lower += particle.v_lower;
        particle.step += particle.v_step;
        particle.step = std::max(std::abs(particle.step), FLT_EPSILON);

        float curr_loss = loss(div, particle.lower, particle.step, data_begin, data_end);
        if (curr_loss < particle.min_loss) {
          particle.min_loss = curr_loss;
          particle.best_lower = particle.lower;
          particle.best_step = particle.step;

          if (curr_loss < global_min_loss) {
            global_min_loss = curr_loss;
            global_best_lower = particle.lower;
            global_best_step = particle.step;
          }
        }
      }
    }

    return {global_best_lower, global_best_lower + global_best_step * div};
  }

  static std::vector<std::pair<float, int>>
  count_freq(const float* data, int size, int d, int dim) {
    std::vector<float> sorted_data;
    sorted_data.reserve(size / dim);
    for (int i = d; i < size; i += dim) {
      sorted_data.push_back(data[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end());

    float curr_value = sorted_data[0];
    int count = 1;
    std::vector<std::pair<float, int>> data_freq_map;
    data_freq_map.reserve(sorted_data.size());
    for (int i = 1; i < sorted_data.size(); i++) {
      if (sorted_data[i] == curr_value) {
        count++;
      } else {
        data_freq_map.emplace_back(curr_value, count);
        curr_value = sorted_data[i];
        count = 1;
      }
    }

    data_freq_map.emplace_back(curr_value, count);
    return data_freq_map;
  }

#if POUQ_X86_64
  using CodeUnit = std::tuple<uint8_t, uint8_t, uint16_t>;
  using RecPara = std::tuple<__m128, __m128>;
#elif POUQ_ARM64
  using CodeUnit = std::tuple<uint8_t, uint8_t, uint16_t>;
  using RecPara = std::tuple<float32x4_t, float32x4_t>;
#else
#error "Unsupported platform"
#endif

#if POUQ_X86_64
  struct AlignedDeleter {
    void
    operator()(void* ptr) const {
      _mm_free(ptr);
    }
  };
#elif POUQ_ARM64
  struct AlignedDeleter {
    void
    operator()(void* ptr) const {
      free(ptr);
    }
  };
#endif

  int dim_ = 0;
  std::unique_ptr<RecPara[], AlignedDeleter> rec_para_ = nullptr;
  std::unique_ptr<CodeUnit[], AlignedDeleter> code_ = nullptr;

  public:
  Quantizer() = default;

  explicit Quantizer(int dim) {
    init(dim);
  }

  void
  init(int dim) {
    assert(dim % 8 == 0);
    dim_ = dim;
  }

  bool
  train(const float* data,
        int size,
        int max_iter = 100,
        int particle_count = 50,
        float scale_factor = 0.1f,
        float init_inertia = 0.9f,
        float final_inertia = 0.4f,
        float init_c1 = 2.5f,
        float final_c1 = 0.5f,
        float init_c2 = 0.5f,
        float final_c2 = 2.5f) {
    assert(code_ == nullptr);
    assert(rec_para_ == nullptr);
    assert(data != nullptr);
    assert(size > 0);
    assert(size % dim_ == 0);

#if POUQ_X86_64
    code_.reset(static_cast<CodeUnit*>(_mm_malloc(size / 8 * sizeof(CodeUnit), 256)));
    if (!code_) {
      return false;
    }

    rec_para_.reset(static_cast<RecPara*>(_mm_malloc(dim_ * 64 * sizeof(RecPara), 256)));
    if (!rec_para_) {
      code_.reset();
      return false;
    }
#elif POUQ_ARM64
    code_.reset(static_cast<CodeUnit*>(aligned_alloc(32, size / 8 * sizeof(CodeUnit))));
    if (!code_) {
      return false;
    }

    rec_para_.reset(static_cast<RecPara*>(aligned_alloc(32, dim_ * 64 * sizeof(RecPara))));
    if (!rec_para_) {
      code_.reset();
      return false;
    }
#endif

    std::vector<float> steps(dim_ * 4);
    std::vector<float> lowers(dim_ * 4);
    std::vector<uint8_t> cid(size);
    std::vector<uint8_t> code(size);

#pragma omp parallel for
    for (int d = 0; d < dim_; d++) {
      auto data_freq_map = count_freq(data, size, d, dim_);
      auto bounds = segment(4, data_freq_map);

      for (int i = 0; i < bounds.size(); i++) {
        auto [lower, upper] = bounds[i];
        if (lower < upper) {
          auto data_begin = std::lower_bound(
              data_freq_map.begin(),
              data_freq_map.end(),
              lower,
              [](const std::pair<float, int>& lhs, float rhs) -> bool { return lhs.first < rhs; });
          auto data_end = std::upper_bound(
              data_freq_map.begin(),
              data_freq_map.end(),
              upper,
              [](float lhs, const std::pair<float, int>& rhs) -> bool { return lhs < rhs.first; });

          std::tie(lower, upper) = optimize(3,
                                            lower,
                                            upper,
                                            data_begin,
                                            data_end,
                                            max_iter,
                                            particle_count,
                                            scale_factor,
                                            init_inertia,
                                            final_inertia,
                                            init_c1,
                                            final_c1,
                                            init_c2,
                                            final_c2);
        }
        lowers[d * 4 + i] = lower;
        if (lower == upper) {
          steps[d * 4 + i] = 1.0;
        } else {
          steps[d * 4 + i] = (upper - lower) / 3.0f;
        }
      }

      for (int i = d; i < size; i += dim_) {
        auto it = std::upper_bound(
            bounds.begin(),
            bounds.end(),
            data[i],
            [](float lhs, const std::pair<float, float>& rhs) -> bool { return lhs < rhs.first; });
        int c = static_cast<int>(it - bounds.begin()) - 1;
        float x = std::clamp((data[i] - lowers[d * 4 + c]) / steps[d * 4 + c] + 0.5f, 0.0f, 3.0f);
        cid[i] = c;
        code[i] = static_cast<uint8_t>(x);
      }
    }

#pragma omp parallel for
    for (int i = 0; i < size / 8; i++) {
      uint8_t x0 = (cid[i * 8] & 3) << 0;
      uint8_t x1 = (cid[i * 8 + 1] & 3) << 2;
      uint8_t x2 = (cid[i * 8 + 2] & 3) << 4;
      uint8_t x3 = (cid[i * 8 + 3] & 3) << 6;
      uint8_t x4 = (cid[i * 8 + 4] & 3) << 0;
      uint8_t x5 = (cid[i * 8 + 5] & 3) << 2;
      uint8_t x6 = (cid[i * 8 + 6] & 3) << 4;
      uint8_t x7 = (cid[i * 8 + 7] & 3) << 6;

      uint16_t x8 = (code[i * 8] & 3) << 0;
      uint16_t x9 = (code[i * 8 + 1] & 3) << 2;
      uint16_t x10 = (code[i * 8 + 2] & 3) << 4;
      uint16_t x11 = (code[i * 8 + 3] & 3) << 6;
      uint16_t x12 = (code[i * 8 + 4] & 3) << 8;
      uint16_t x13 = (code[i * 8 + 5] & 3) << 10;
      uint16_t x14 = (code[i * 8 + 6] & 3) << 12;
      uint16_t x15 = (code[i * 8 + 7] & 3) << 14;

      code_.get()[i] = std::make_tuple(
          x0 | x1 | x2 | x3, x4 | x5 | x6 | x7, x8 | x9 | x10 | x11 | x12 | x13 | x14 | x15);
    }

#pragma omp parallel for
    for (int g = 0; g < dim_ / 4; g++) {
      for (int j = 0; j < 256; j++) {
        int x0 = g * 16 + 0 * 4 + (j & 3);
        int x1 = g * 16 + 1 * 4 + (j >> 2 & 3);
        int x2 = g * 16 + 2 * 4 + (j >> 4 & 3);
        int x3 = g * 16 + 3 * 4 + (j >> 6 & 3);

#if POUQ_X86_64
        rec_para_[g * 256 + j] = {
            _mm_setr_ps(lowers[x0], lowers[x1], lowers[x2], lowers[x3]),
            _mm_setr_ps(steps[x0], steps[x1], steps[x2], steps[x3]),
        };
#elif POUQ_ARM64
        float lower_array[4] = {lowers[x0], lowers[x1], lowers[x2], lowers[x3]};
        float step_array[4] = {steps[x0], steps[x1], steps[x2], steps[x3]};
        rec_para_[g * 256 + j] = {
            vld1q_f32(lower_array),
            vld1q_f32(step_array),
        };
#endif
      }
    }

    return true;
  }

  float
  distance(const float* data, int offset) {
    assert(data != nullptr);
    assert(offset % dim_ == 0);

#if POUQ_X86_64
    __m256 sum_squares_vec = _mm256_setzero_ps();
    for (int dim = 0; dim < dim_; dim += 8) {
      int group_idx = dim / 4;
      auto [code1, code2, code_value] = code_.get()[(offset / 4 + group_idx) / 2];
      auto [lower1, step1] = rec_para_.get()[group_idx * 256 + code1];
      auto [lower2, step2] = rec_para_.get()[(group_idx + 1) * 256 + code2];

      __m256 lower_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(lower1), lower2, 1);
      __m256 step_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(step1), step2, 1);

      __m256i code_bytes = _mm256_set1_epi32(code_value);
      __m256i shift_amounts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
      __m256i shifted_code = _mm256_srlv_epi32(code_bytes, shift_amounts);
      __m256i mask = _mm256_set1_epi32(3);
      __m256i masked_code = _mm256_and_si256(shifted_code, mask);

      __m256 code_vec = _mm256_cvtepi32_ps(masked_code);
      __m256 reconstructed_vec = _mm256_fmadd_ps(code_vec, step_vec, lower_vec);

      __m256 data_vec = _mm256_loadu_ps(data + dim);
      __m256 diff_vec = _mm256_sub_ps(reconstructed_vec, data_vec);
      sum_squares_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_squares_vec);
    }

    __m128 sum_low128 = _mm256_castps256_ps128(sum_squares_vec);
    __m128 sum_high128 = _mm256_extractf128_ps(sum_squares_vec, 1);
    __m128 total_sum128 = _mm_add_ps(sum_low128, sum_high128);

    __m128 shuffled_sum = _mm_movehdup_ps(total_sum128);
    total_sum128 = _mm_add_ps(total_sum128, shuffled_sum);
    shuffled_sum = _mm_movehl_ps(shuffled_sum, total_sum128);
    total_sum128 = _mm_add_ss(total_sum128, shuffled_sum);

    return _mm_cvtss_f32(total_sum128);

#elif POUQ_ARM64
    float sum_squares = 0.0f;
    for (int dim = 0; dim < dim_; dim += 8) {
      int group_idx = dim / 4;
      auto [code1, code2, code_value] = code_.get()[(offset / 4 + group_idx) / 2];
      auto [lower1, step1] = rec_para_.get()[group_idx * 256 + code1];
      auto [lower2, step2] = rec_para_.get()[(group_idx + 1) * 256 + code2];

      uint32x4_t code_uint1 = vsetq_lane_u32(code_value & 3, vdupq_n_u32(0), 0);
      code_uint1 = vsetq_lane_u32((code_value >> 2) & 3, code_uint1, 1);
      code_uint1 = vsetq_lane_u32((code_value >> 4) & 3, code_uint1, 2);
      code_uint1 = vsetq_lane_u32((code_value >> 6) & 3, code_uint1, 3);

      uint32x4_t code_uint2 = vsetq_lane_u32((code_value >> 8) & 3, vdupq_n_u32(0), 0);
      code_uint2 = vsetq_lane_u32((code_value >> 10) & 3, code_uint2, 1);
      code_uint2 = vsetq_lane_u32((code_value >> 12) & 3, code_uint2, 2);
      code_uint2 = vsetq_lane_u32((code_value >> 14) & 3, code_uint2, 3);

      float32x4_t code_vec1 = vcvtq_f32_u32(code_uint1);
      float32x4_t reconstructed_vec1 = vmlaq_f32(lower1, code_vec1, step1);
      float32x4_t data_vec1 = vld1q_f32(data + dim);
      float32x4_t diff_vec1 = vsubq_f32(reconstructed_vec1, data_vec1);
      float32x4_t sq_diff1 = vmulq_f32(diff_vec1, diff_vec1);

      float32x4_t code_vec2 = vcvtq_f32_u32(code_uint2);
      float32x4_t reconstructed_vec2 = vmlaq_f32(lower2, code_vec2, step2);
      float32x4_t data_vec2 = vld1q_f32(data + dim + 4);
      float32x4_t diff_vec2 = vsubq_f32(reconstructed_vec2, data_vec2);
      float32x4_t sq_diff2 = vmulq_f32(diff_vec2, diff_vec2);

      float32x4_t combined = vaddq_f32(sq_diff1, sq_diff2);
      sum_squares += vaddvq_f32(combined);
    }

    return sum_squares;
#endif
  }
};

}  // namespace pouq