#pragma once

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cfloat>

namespace pouq {

class Quantizer final {
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
       float step_size,
       const std::vector<std::pair<float, int>>::iterator& data_begin,
       const std::vector<std::pair<float, int>>::iterator& data_end) {
    assert(div > 0.0f);
    assert(step_size >= FLT_EPSILON);
    assert(data_begin <= data_end);

    float total_loss = 0.0f;

    for (auto it = data_begin; it != data_end; ++it) {
      auto& [data_value, point_count] = *it;
      float real_quantized_code = (data_value - lower) / step_size;
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

    return total_loss * step_size * step_size;
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
    float init_step_size = init_range_width / div;

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
          : lower(l_val), step_size(s_val), v_lower(vl_val), v_step_size(vs_val) {
        best_lower = lower;
        best_step_size = step_size;
        min_loss = FLT_MAX;
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
      float curr_loss = loss(div, particle.lower, particle.step_size, data_begin, data_end);

      particle.min_loss = curr_loss;
      if (curr_loss < global_min_loss) {
        global_min_loss = curr_loss;
        global_best_lower = particle.lower;
        global_best_step_size = particle.step_size;
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

        particle.v_step_size = inertia * particle.v_step_size +
                               c1 * r1 * (particle.best_step_size - particle.step_size) +
                               c2 * r2 * (global_best_step_size - particle.step_size);

        particle.lower += particle.v_lower;
        particle.step_size += particle.v_step_size;

        particle.step_size = std::max(std::abs(particle.step_size), FLT_EPSILON);

        float curr_loss = loss(div, particle.lower, particle.step_size, data_begin, data_end);

        if (curr_loss < particle.min_loss) {
          particle.min_loss = curr_loss;
          particle.best_lower = particle.lower;
          particle.best_step_size = particle.step_size;

          if (curr_loss < global_min_loss) {
            global_min_loss = curr_loss;
            global_best_lower = particle.lower;
            global_best_step_size = particle.step_size;
          }
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
    assert(dim % 8 == 0);
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
    assert(code_ == nullptr);
    assert(rec_para_ == nullptr);
    assert(data != nullptr);
    assert(size > 0);
    assert(size % dim_ == 0);

    code_ = static_cast<CodeUnit*>(_mm_malloc(size / 4 * sizeof(CodeUnit), 256));
    if (!code_) {
      throw std::bad_alloc();
    }

    rec_para_ = static_cast<RecPara*>(_mm_malloc(dim_ * 64 * sizeof(RecPara), 256));
    if (!rec_para_) {
      _mm_free(code_);
      code_ = nullptr;
      throw std::bad_alloc();
    }

    std::vector<float> step_size(dim_ * 4);
    std::vector<float> lower_bound(dim_ * 4);
    std::vector<uint8_t> cid(size / 4);
    std::vector<uint16_t> code(size / 8);

#pragma omp parallel for
    for (int d = 0; d < dim_; d++) {
      auto data_freq_map = count_freq(data, size, d);
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
        lower_bound[d * 4 + i] = lower;
        if (lower == upper) {
          step_size[d * 4 + i] = 1.0;
        } else {
          step_size[d * 4 + i] = (upper - lower) / 3.0f;
        }
      }

      for (int i = d; i < size; i += dim_) {
        auto it = std::upper_bound(
            bounds.begin(),
            bounds.end(),
            data[i],
            [](float lhs, const std::pair<float, float>& rhs) -> bool { return lhs < rhs.first; });
        int c = static_cast<int>(it - bounds.begin()) - 1;
        float x = std::clamp(
            (data[i] - lower_bound[d * 4 + c]) / step_size[d * 4 + c] + 0.5f, 0.0f, 3.0f);
        set8(&cid[i / dim_ * dim_ / 4], i % dim_, c);
        set16(&code[i / dim_ * dim_ / 8], i % dim_, x);
      }
    }

    for (int i = 0; i < size / 4; i += 2) {
      code_[i / 2] = std::make_tuple(cid[i], cid[i + 1], code[i / 2]);
    }

#pragma omp parallel for
    for (int g = 0; g < dim_ / 4; g++) {
      for (int j = 0; j < 256; j++) {
        auto [x0, x1, x2, x3] = get8(j);
        __m128 lb = _mm_setr_ps(lower_bound[g * 16 + 0 * 4 + x0],
                                lower_bound[g * 16 + 1 * 4 + x1],
                                lower_bound[g * 16 + 2 * 4 + x2],
                                lower_bound[g * 16 + 3 * 4 + x3]);
        __m128 st = _mm_setr_ps(step_size[g * 16 + 0 * 4 + x0],
                                step_size[g * 16 + 1 * 4 + x1],
                                step_size[g * 16 + 2 * 4 + x2],
                                step_size[g * 16 + 3 * 4 + x3]);
        rec_para_[g * 256 + j] = {lb, st};
      }
    }
  }

  float
  l2distance(const float* data, int offset) {
    assert(data != nullptr);
    assert(offset % dim_ == 0);

    offset /= 4;
    __m256 sum_vec = _mm256_setzero_ps();

    static __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    static __m256i mask = _mm256_set1_epi32(3);

    for (int i = 0; i < dim_; i += 8) {
      int idx = i / 4;
      auto [c1, c2, code] = code_[(offset + idx) / 2];
      auto [lb1, st1] = rec_para_[idx * 256 + c1];
      auto [lb2, st2] = rec_para_[(idx + 1) * 256 + c2];
      __m256 lb_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(lb1), lb2, 1);
      __m256 st_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(st1), st2, 1);
      __m256i bytes = _mm256_set1_epi32(code);
      __m256i shifted = _mm256_srlv_epi32(bytes, shifts);
      __m256i masked = _mm256_and_si256(shifted, mask);
      __m256 code_vec = _mm256_cvtepi32_ps(masked);
      __m256 reconstructed = _mm256_fmadd_ps(code_vec, st_vec, lb_vec);
      __m256 data_vec = _mm256_loadu_ps(data + i);
      __m256 diff = _mm256_sub_ps(reconstructed, data_vec);
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
    assert(code_ != nullptr);
    assert(rec_para_ != nullptr);
    _mm_free(code_);
    _mm_free(rec_para_);
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
  set8(uint8_t* data, int i, int n) {
    int offset = (i & 3) << 1;
    i >>= 2;
    data[i] &= ~(3 << offset);
    data[i] |= n << offset;
  }

  static std::tuple<int, int, int, int>
  get8(uint8_t byte) {
    return {
        byte & 3,
        byte >> 2 & 3,
        byte >> 4 & 3,
        byte >> 6 & 3,
    };
  }

  static void
  set16(uint16_t* data, int i, int n) {
    int offset = (i & 7) << 1;
    i >>= 3;
    data[i] &= ~(3 << offset);
    data[i] |= n << offset;
  }
};

}  // namespace pouq