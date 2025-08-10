#pragma once

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace pouq {

class Quantizer final {
  static std::vector<std::pair<float, float>>
  segment(int k, const std::vector<std::pair<float, int>>& data_freq_map) {
    const int size = data_freq_map.size();
    k = std::min(size, k);

    std::vector sum_count(size + 1, 0.0);
    for (int i = 1; i <= size; ++i) {
      sum_count[i] = sum_count[i - 1] + static_cast<float>(data_freq_map[i - 1].second);
    }

    std::vector prev_dp(size + 1, std::numeric_limits<float>::max());
    std::vector curr_dp(size + 1, std::numeric_limits<float>::max());
    std::vector prev_idx(size + 1, std::vector<int>(k + 1, 0));
    prev_dp[0] = 0.0;

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

        const int mid = (l + r) / 2;
        const int start = std::max(j - 1, opt_l);
        const int end = std::min(mid - 1, opt_r);
        float min_cost = std::numeric_limits<float>::max();
        int split_pos = 0;
        for (int m = start; m <= end; ++m) {
          const float width = static_cast<float>(data_freq_map[mid - 1].first) -
                              static_cast<float>(data_freq_map[m].first);
          const float count = sum_count[mid] - sum_count[m];
          const float cost = prev_dp[m] + width * width * count;
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
      std::fill(curr_dp.begin(), curr_dp.end(), std::numeric_limits<float>::max());
    }

    std::vector<int> split_pos(k);
    int curr_pos = size;
    for (int j = k; j > 0; --j) {
      const int m = prev_idx[curr_pos][j];
      split_pos[j - 1] = m;
      curr_pos = m;
    }

    std::vector<std::pair<float, float>> bounds(k);
    for (int t = 0; t < k; ++t) {
      const int start = split_pos[t];
      const int end = t < k - 1 ? split_pos[t + 1] - 1 : size - 1;
      bounds[t] = {data_freq_map[start].first, data_freq_map[end].first};
    }

    return bounds;
  }

  static float
  loss(float div,
       float lower,
       float step_size,
       const std::vector<std::pair<float, int>>::const_iterator& data_begin,
       const std::vector<std::pair<float, int>>::const_iterator& data_end) {
    float total_loss = 0.0f;

    for (auto it = data_begin; it != data_end; ++it) {
      const auto& [data_value, point_count] = *it;
      const float real_quantized_code = (data_value - lower) / step_size;
      float quantized_code = 0.0f;

      if (data_value > lower) {
        quantized_code = std::round(real_quantized_code);
        if (quantized_code > div) {
          quantized_code = div;
        }
      }

      const float code_loss = real_quantized_code - quantized_code;
      total_loss += code_loss * code_loss * static_cast<float>(point_count);
    }

    return total_loss * step_size * step_size;
  }

  static std::pair<float, float>
  optimize(float div,
           float init_lower,
           float init_upper,
           const std::vector<std::pair<float, int>>::const_iterator& data_begin,
           const std::vector<std::pair<float, int>>::const_iterator& data_end,
           int max_iter,
           int particle_count,
           float scale_factor,
           float init_inertia,
           float final_inertia,
           float init_c1,
           float final_c1,
           float init_c2,
           float final_c2) {
    const float init_range_width = init_upper - init_lower;
    const float init_step_size = init_range_width / div;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution v_dis(-init_range_width * 0.1f, init_range_width * 0.1f);
    std::uniform_real_distribution p_dis(0.0f, 1.0f);
    std::uniform_real_distribution lower_dis(init_lower - init_range_width * scale_factor,
                                             init_lower + init_range_width * scale_factor);
    std::uniform_real_distribution step_dis(init_step_size * (1.0f - scale_factor),
                                            init_step_size * (1.0f + scale_factor));

    struct Particle {
      float lower;
      float step_size;
      float v_lower;
      float v_step_size;
      float best_lower;
      float best_step_size;
      float min_loss;

      Particle(float l_val, float s_val, float vl_val, float vs_val)
          : lower(l_val),
            step_size(s_val),
            v_lower(vl_val),
            v_step_size(vs_val),
            best_lower(l_val),
            best_step_size(s_val),
            min_loss(std::numeric_limits<float>::max()) {
      }
    };

    std::vector<Particle> swarm;
    swarm.reserve(particle_count);
    for (int i = 0; i < particle_count; i++) {
      swarm.emplace_back(lower_dis(gen), step_dis(gen), v_dis(gen), v_dis(gen));
    }

    float global_best_lower = init_lower;
    float global_best_step_size = init_step_size;
    float global_min_loss = loss(div, init_lower, init_step_size, data_begin, data_end);

    for (auto& particle : swarm) {
      const float curr_loss = loss(div, particle.lower, particle.step_size, data_begin, data_end);

      particle.min_loss = curr_loss;
      if (curr_loss < global_min_loss) {
        global_min_loss = curr_loss;
        global_best_lower = particle.lower;
        global_best_step_size = particle.step_size;
      }
    }

    for (int iter = 0; iter < max_iter; ++iter) {
      const float x = static_cast<float>(iter) / max_iter;
      const float inertia = init_inertia - (init_inertia - final_inertia) * x;
      const float c1 = init_c1 - (init_c1 - final_c1) * x;
      const float c2 = init_c2 - (init_c2 - final_c2) * x;

      for (auto& particle : swarm) {
        const float r1 = p_dis(gen);
        const float r2 = p_dis(gen);

        particle.v_lower = inertia * particle.v_lower +
                           c1 * r1 * (particle.best_lower - particle.lower) +
                           c2 * r2 * (global_best_lower - particle.lower);

        particle.v_step_size = inertia * particle.v_step_size +
                               c1 * r1 * (particle.best_step_size - particle.step_size) +
                               c2 * r2 * (global_best_step_size - particle.step_size);

        particle.lower += particle.v_lower;
        particle.step_size += particle.v_step_size;

        if (particle.step_size <= std::numeric_limits<float>::epsilon()) {
          particle.step_size = std::numeric_limits<float>::epsilon();
        }

        const float curr_loss = loss(div, particle.lower, particle.step_size, data_begin, data_end);

        if (curr_loss < particle.min_loss) {
          particle.min_loss = curr_loss;
          particle.best_lower = particle.lower;
          particle.best_step_size = particle.step_size;
        }

        if (curr_loss < global_min_loss) {
          global_min_loss = curr_loss;
          global_best_lower = particle.lower;
          global_best_step_size = particle.step_size;
        }
      }
    }

    return {global_best_lower, global_best_lower + global_best_step_size * div};
  }

  using CodeUnit = std::tuple<uint8_t, uint8_t, uint16_t>;
  using RecPara = std::tuple<__m128, __m128>;

  public:
  Quantizer() = default;

  explicit Quantizer(int dim) {
    init(dim);
  }

  void
  init(int dim) {
    dim_ = dim;
  }

  void
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
    if (code_) {
      _mm_free(code_);
      code_ = nullptr;
    }
    if (rec_para_) {
      _mm_free(rec_para_);
      rec_para_ = nullptr;
    }

    int combined_data_size = size / 4;
    code_ = static_cast<CodeUnit*>(_mm_malloc(combined_data_size * sizeof(CodeUnit), 256));
    if (!code_) {
      throw std::bad_alloc();
    }

    int bounds_data_size = dim_ * 64;
    rec_para_ = static_cast<RecPara*>(_mm_malloc(bounds_data_size * sizeof(RecPara), 256));
    if (!rec_para_) {
      _mm_free(code_);
      code_ = nullptr;
      throw std::bad_alloc();
    }

    std::vector<float> step_size(dim_ * 4);
    std::vector<float> lower_bound(dim_ * 4);
    std::vector<uint8_t> cid(size / 4);
    std::vector<uint16_t> code(size / 8);

    const int dim_div_4 = dim_ / 4;

#pragma omp parallel for
    for (int d = 0; d < dim_; d++) {
      const auto data_freq_map = count_freq(data, size, d);
      const auto bounds = segment(4, data_freq_map);
      const int d_times_4 = d * 4;

      for (int i = 0; i < bounds.size(); i++) {
        auto [lower, upper] = bounds[i];
        if (lower < upper) {
          const auto data_begin =
              std::lower_bound(data_freq_map.begin(),
                               data_freq_map.end(),
                               lower,
                               [](const std::pair<float, int>& lhs, const float rhs) -> bool {
                                 return lhs.first < rhs;
                               });
          const auto data_end =
              std::upper_bound(data_freq_map.begin(),
                               data_freq_map.end(),
                               upper,
                               [](const float rhs, const std::pair<float, int>& lhs) -> bool {
                                 return rhs < lhs.first;
                               });

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
        lower_bound[d_times_4 + i] = lower;
        if (lower == upper) {
          step_size[d_times_4 + i] = 1.0;
        } else {
          step_size[d_times_4 + i] = (upper - lower) / 3.0f;
        }
      }

      for (int i = d; i < size; i += dim_) {
        const auto it = std::upper_bound(
            bounds.begin(),
            bounds.end(),
            data[i],
            [](float rhs, const std::pair<float, float>& lhs) -> bool { return rhs < lhs.first; });
        const int c = it - bounds.begin() - 1;
        const float x = std::clamp(
            (data[i] - lower_bound[d_times_4 + c]) / step_size[d_times_4 + c] + 0.5f, 0.0f, 3.0f);
        const int base_index = (i / dim_) * dim_div_4;
        set(&cid[base_index], i % dim_, c);
        set16(&code[base_index / 2], i % dim_, x);
      }
    }

    for (int i = 0; i < size / 4; i += 2) {
      code_[i / 2] = std::make_tuple(cid[i], cid[i + 1], code[i / 2]);
    }

#pragma omp parallel for
    for (int g = 0; g < dim_ / 4; g++) {
      for (int j = 0; j < 256; j++) {
        const auto [x0, x1, x2, x3] = get(j);
        const int base_idx = g * 16;
        const __m128 lb = _mm_setr_ps(lower_bound[base_idx + 0 * 4 + x0],
                                      lower_bound[base_idx + 1 * 4 + x1],
                                      lower_bound[base_idx + 2 * 4 + x2],
                                      lower_bound[base_idx + 3 * 4 + x3]);
        const __m128 st = _mm_setr_ps(step_size[base_idx + 0 * 4 + x0],
                                      step_size[base_idx + 1 * 4 + x1],
                                      step_size[base_idx + 2 * 4 + x2],
                                      step_size[base_idx + 3 * 4 + x3]);
        rec_para_[g * 256 + j] = {lb, st};
      }
    }
  }

  float
  l2distance(const float* data, int offset) const {
    offset /= 4;
    __m256 sum_vec = _mm256_setzero_ps();

    static const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    static const __m256i mask = _mm256_set1_epi32(3);

    for (int i = 0; i < dim_; i += 8) {
      const int idx = i / 4;
      const auto [c1, c2, code] = code_[(offset + idx) / 2];
      const auto [lb1, st1] = rec_para_[idx * 256 + c1];
      const auto [lb2, st2] = rec_para_[(idx + 1) * 256 + c2];
      const __m256 lb_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(lb1), lb2, 1);
      const __m256 st_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(st1), st2, 1);
      const __m256i bytes = _mm256_set1_epi32(code);
      const __m256i shifted = _mm256_srlv_epi32(bytes, shifts);
      const __m256i masked = _mm256_and_si256(shifted, mask);
      const __m256 code_vec = _mm256_cvtepi32_ps(masked);
      const __m256 reconstructed = _mm256_fmadd_ps(code_vec, st_vec, lb_vec);
      const __m256 data_vec = _mm256_loadu_ps(data + i);
      const __m256 diff = _mm256_sub_ps(reconstructed, data_vec);
      sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    __m128 low128 = _mm256_castps256_ps128(sum_vec);
    __m128 high128 = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);
    __m128 shuf = _mm_movehdup_ps(sum128);
    sum128 = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sum128);
    sum128 = _mm_add_ss(sum128, shuf);
    return _mm_cvtss_f32(sum128);
  }

  ~Quantizer() {
    if (code_) {
      _mm_free(code_);
    }
    if (rec_para_) {
      _mm_free(rec_para_);
    }
  }

  private:
  int dim_ = 0;
  RecPara* rec_para_ = nullptr;
  CodeUnit* code_ = nullptr;

  std::vector<std::pair<float, int>>
  count_freq(const float* data, int size, int group) const {
    std::vector<float> sorted_data;
    sorted_data.reserve(size / dim_);
    for (int i = group; i < size; i += dim_) {
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

  static void
  set(uint8_t* data, int i, int n) {
    const int offset = (i & 3) << 1;
    i >>= 2;
    data[i] &= ~(3 << offset);
    data[i] |= n << offset;
  }

  static std::tuple<int, int, int, int>
  get(uint8_t byte) {
    return {
        byte & 3,
        byte >> 2 & 3,
        byte >> 4 & 3,
        byte >> 6 & 3,
    };
  }

  static void
  set16(uint16_t* data, int i, int n) {
    const int offset = (i & 7) << 1;
    i >>= 3;
    data[i] &= ~(3 << offset);
    data[i] |= n << offset;
  }
};

}  // namespace pouq