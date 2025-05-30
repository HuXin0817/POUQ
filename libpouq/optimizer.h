#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace pouq {

class Optimizer {
private:
  static constexpr uint64_t max_iter          = 128;
  static constexpr uint64_t grid_side_length  = 8;
  static constexpr float    grid_scale_factor = 0.1f;
  static constexpr float    initial_inertia   = 0.9f;
  static constexpr float    final_inertia     = 0.4f;
  static constexpr float    c1                = 1.8f;
  static constexpr float    c2                = 1.8f;

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

  static float loss(const float                                      div,
      float                                                          cluster_lower_bound,
      float                                                          step_size_value,
      const std::vector<std::pair<float, uint64_t>>::const_iterator &data_begin,
      const std::vector<std::pair<float, uint64_t>>::const_iterator &data_end) {
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

public:
  std::pair<float, float> operator()(const float                     div,
      const float                                                    initial_min_bound,
      const float                                                    initial_max_bound,
      const std::vector<std::pair<float, uint64_t>>::const_iterator &data_start,
      const std::vector<std::pair<float, uint64_t>>::const_iterator &data_end) {
    const float initial_range_width  = initial_max_bound - initial_min_bound;
    const float initial_range_center = (initial_min_bound + initial_max_bound) * 0.5f;

    std::random_device             rd;
    std::mt19937                   gen(rd());
    std::uniform_real_distribution v_dis(-initial_range_width * 0.1f, initial_range_width * 0.1f);
    std::uniform_real_distribution p_dis(0.0f, 1.0f);

    std::vector<Particle> particle_swarm;

    for (uint64_t i = 0; i < grid_side_length; i++) {
      for (uint64_t j = 0; j < grid_side_length; j++) {
        const float lower_bound =
            initial_min_bound - grid_scale_factor * initial_range_width +
            static_cast<float>(i) * 2 * grid_scale_factor * initial_range_width / grid_side_length;
        const float upper_bound =
            initial_max_bound - grid_scale_factor * initial_range_width +
            static_cast<float>(j) * 2 * grid_scale_factor * initial_range_width / grid_side_length;
        const float particle_center = (lower_bound + upper_bound) / 2.0f;
        const float particle_width  = upper_bound - lower_bound;

        particle_swarm.emplace_back(particle_center, particle_width, v_dis(gen), v_dis(gen));
      }
    }

    float global_best_center = initial_range_center;
    float global_best_width  = initial_range_width;
    float global_min_loss    = loss(div, initial_min_bound, initial_range_width / div, data_start, data_end);

    for (auto &particle : particle_swarm) {
      const float current_lower_bound = particle.center - particle.width * 0.5f;
      const float current_loss        = loss(div, current_lower_bound, particle.width / div, data_start, data_end);

      particle.min_loss = current_loss;
      if (current_loss < global_min_loss) {
        global_min_loss    = current_loss;
        global_best_center = particle.center;
        global_best_width  = particle.width;
      }
    }

    for (uint64_t iter = 0; iter < max_iter; ++iter) {
      const float current_inertia =
          initial_inertia - (initial_inertia - final_inertia) * static_cast<float>(iter) / max_iter;

      for (auto &particle : particle_swarm) {
        const float r1 = p_dis(gen);
        const float r2 = p_dis(gen);

        particle.v_center = current_inertia * particle.v_center + c1 * r1 * (particle.best_center - particle.center) +
                            c2 * r2 * (global_best_center - particle.center);

        particle.v_width = current_inertia * particle.v_width + c1 * r1 * (particle.best_width - particle.width) +
                           c2 * r2 * (global_best_width - particle.width);

        particle.center += particle.v_center;
        particle.width += particle.v_width;

        if (particle.width <= std::numeric_limits<float>::epsilon()) {
          particle.width = std::numeric_limits<float>::epsilon();
        }

        const float current_lower = particle.center - particle.width * 0.5f;
        const float current_loss  = loss(div, current_lower, particle.width / div, data_start, data_end);

        if (current_loss < particle.min_loss) {
          particle.min_loss    = current_loss;
          particle.best_center = particle.center;
          particle.best_width  = particle.width;
        }

        if (current_loss < global_min_loss) {
          global_min_loss    = current_loss;
          global_best_center = particle.center;
          global_best_width  = particle.width;
        }
      }
    }

    const float opt_lower = global_best_center - global_best_width * 0.5f;
    const float opt_upper = global_best_center + global_best_width * 0.5f;
    return {opt_lower, opt_upper};
  }
};

}  // namespace pouq