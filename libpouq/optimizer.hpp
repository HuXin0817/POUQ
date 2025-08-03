#pragma once

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

namespace pouq {

static constexpr size_t max_iter          = 128;
static constexpr size_t grid_side_length  = 8;
static constexpr float  grid_scale_factor = 0.1f;
static constexpr float  init_inertia      = 0.9f;
static constexpr float  final_inertia     = 0.4f;
static constexpr float  c1                = 1.8f;
static constexpr float  c2                = 1.8f;

struct Particle {
  float center;
  float width;
  float v_center;
  float v_width;
  float best_center;
  float best_width;
  float min_loss;

  Particle(const float c_val, const float w_val, const float vc_val, const float vw_val)
      : center(c_val),
        width(w_val),
        v_center(vc_val),
        v_width(vw_val),
        best_center(c_val),
        best_width(w_val),
        min_loss(std::numeric_limits<float>::max()) {}
};

static float loss(const float                                    div,
    float                                                        cluster_lower_bound,
    float                                                        step_size_value,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_end) {
  step_size_value  = std::max(step_size_value, 1e-8f);
  float total_loss = 0.0f;

  for (auto it = data_begin; it != data_end; ++it) {
    const auto &[data_value, point_count] = *it;
    const float real_quantized_code       = (data_value - cluster_lower_bound) / step_size_value;
    float       quantized_code            = 0.0f;

    if (data_value > cluster_lower_bound) {
      quantized_code = std::round(real_quantized_code);
      if (quantized_code > div) {
        quantized_code = div;
      }
    }

    const float code_loss = real_quantized_code - quantized_code;
    total_loss += code_loss * code_loss * static_cast<float>(point_count);
  }

  return total_loss * step_size_value * step_size_value;
}

std::pair<float, float> optimize(float                           div,
    float                                                        init_lower_bound,
    float                                                        init_upper_bound,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_start,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_end) {
  const float init_range_width  = init_upper_bound - init_lower_bound;
  const float init_range_center = (init_lower_bound + init_upper_bound) * 0.5f;

  std::random_device             rd;
  std::mt19937                   gen(rd());
  std::uniform_real_distribution v_dis(-init_range_width * 0.1f, init_range_width * 0.1f);
  std::uniform_real_distribution p_dis(0.0f, 1.0f);

  std::vector<Particle> swarm;
  swarm.reserve(grid_side_length * grid_side_length);
  for (size_t i = 0; i < grid_side_length; i++) {
    for (size_t j = 0; j < grid_side_length; j++) {
      const float lower_bound = init_lower_bound - grid_scale_factor * init_range_width +
                                static_cast<float>(i) * 2 * grid_scale_factor * init_range_width / grid_side_length;
      const float upper_bound = init_upper_bound - grid_scale_factor * init_range_width +
                                static_cast<float>(j) * 2 * grid_scale_factor * init_range_width / grid_side_length;
      const float particle_center = (lower_bound + upper_bound) / 2.0f;
      const float particle_width  = upper_bound - lower_bound;

      swarm.emplace_back(particle_center, particle_width, v_dis(gen), v_dis(gen));
    }
  }

  float global_best_center = init_range_center;
  float global_best_width  = init_range_width;
  float global_min_loss    = loss(div, init_lower_bound, init_range_width / div, data_start, data_end);

  for (auto &particle : swarm) {
    const float curr_lower_bound = particle.center - particle.width * 0.5f;
    const float curr_loss        = loss(div, curr_lower_bound, particle.width / div, data_start, data_end);

    particle.min_loss = curr_loss;
    if (curr_loss < global_min_loss) {
      global_min_loss    = curr_loss;
      global_best_center = particle.center;
      global_best_width  = particle.width;
    }
  }

  for (size_t iter = 0; iter < max_iter; ++iter) {
    const float inertia = init_inertia - (init_inertia - final_inertia) * static_cast<float>(iter) / max_iter;

    for (auto &particle : swarm) {
      const float r1 = p_dis(gen);
      const float r2 = p_dis(gen);

      particle.v_center = inertia * particle.v_center + c1 * r1 * (particle.best_center - particle.center) +
                          c2 * r2 * (global_best_center - particle.center);

      particle.v_width = inertia * particle.v_width + c1 * r1 * (particle.best_width - particle.width) +
                         c2 * r2 * (global_best_width - particle.width);

      particle.center += particle.v_center;
      particle.width += particle.v_width;

      if (particle.width <= std::numeric_limits<float>::epsilon()) {
        particle.width = std::numeric_limits<float>::epsilon();
      }

      const float curr_lower = particle.center - particle.width * 0.5f;
      const float curr_loss  = loss(div, curr_lower, particle.width / div, data_start, data_end);

      if (curr_loss < particle.min_loss) {
        particle.min_loss    = curr_loss;
        particle.best_center = particle.center;
        particle.best_width  = particle.width;
      }

      if (curr_loss < global_min_loss) {
        global_min_loss    = curr_loss;
        global_best_center = particle.center;
        global_best_width  = particle.width;
      }
    }
  }

  const float opt_lower = global_best_center - global_best_width * 0.5f;
  const float opt_upper = global_best_center + global_best_width * 0.5f;
  return {opt_lower, opt_upper};
}

}  // namespace pouq
