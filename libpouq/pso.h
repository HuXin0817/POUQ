#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

struct Particle {
  float center;
  float width;
  float velocity_center;
  float velocity_width;
  float best_center;
  float best_width;
  float min_loss_value;

  Particle(float c_val, float w_val, float vc_val, float vw_val)
      : center(c_val),
        width(w_val),
        velocity_center(vc_val),
        velocity_width(vw_val),
        best_center(c_val),
        best_width(w_val),
        min_loss_value(std::numeric_limits<float>::max()) {}
};

float loss(const float                                             div,
    float                                                          cluster_lower_bound,
    float                                                          step_size_value,
    const std::vector<std::pair<float, uint64_t>>::const_iterator &data_begin,
    const std::vector<std::pair<float, uint64_t>>::const_iterator &data_end) {
  step_size_value   = std::max(step_size_value, 1e-8f);
  float total_error = 0.0f;

  for (auto it = data_begin; it != data_end; ++it) {
    const auto &[data_value, point_count] = *it;
    float real_quantized_code             = (data_value - cluster_lower_bound) / step_size_value;
    float quantized_code                  = 0.0f;

    if (data_value > cluster_lower_bound) {
      quantized_code = std::round(real_quantized_code);
      if (quantized_code > div) {
        quantized_code = div;
      }
    }

    float code_error = real_quantized_code - quantized_code;
    total_error += code_error * code_error * static_cast<float>(point_count);
  }

  return total_error * step_size_value * step_size_value;
}

std::pair<float, float> pso_optimize(const float                   div,
    const float                                                    initial_min_bound,
    const float                                                    initial_max_bound,
    const std::vector<std::pair<float, uint64_t>>::const_iterator &data_start,
    const std::vector<std::pair<float, uint64_t>>::const_iterator &data_end,
    const uint64_t                                                 max_iterations,
    const uint64_t                                                 grid_side_length,
    const float                                                    grid_scale_factor,
    const float                                                    initial_inertia,
    const float                                                    final_inertia,
    const float                                                    c1,
    const float                                                    c2) {
  const float initial_range_width  = initial_max_bound - initial_min_bound;
  const float initial_range_center = (initial_min_bound + initial_max_bound) * 0.5f;

  std::random_device                    random_device;
  std::mt19937                          random_engine(random_device());
  std::uniform_real_distribution<float> velocity_distribution(-initial_range_width * 0.1f, initial_range_width * 0.1f);
  std::uniform_real_distribution<float> probability_distribution(0.0f, 1.0f);

  std::vector<Particle> particle_swarm;

  for (uint64_t i = 0; i < grid_side_length; i++) {
    for (uint64_t j = 0; j < grid_side_length; j++) {
      float lower_bound = initial_min_bound - grid_scale_factor * initial_range_width +
                          static_cast<float>(i) * 2 * grid_scale_factor * initial_range_width / grid_side_length;
      float upper_bound = initial_max_bound - grid_scale_factor * initial_range_width +
                          static_cast<float>(j) * 2 * grid_scale_factor * initial_range_width / grid_side_length;
      float particle_center = (lower_bound + upper_bound) / 2.0f;
      float particle_width  = upper_bound - lower_bound;

      particle_swarm.emplace_back(
          particle_center, particle_width, velocity_distribution(random_engine), velocity_distribution(random_engine));
    }
  }

  float global_best_center = initial_range_center;
  float global_best_width  = initial_range_width;
  float global_min_loss    = loss(div, initial_min_bound, initial_range_width / div, data_start, data_end);

  for (auto &particle : particle_swarm) {
    float current_lower_bound = particle.center - particle.width * 0.5f;
    float current_loss        = loss(div, current_lower_bound, particle.width / div, data_start, data_end);

    particle.min_loss_value = current_loss;
    if (current_loss < global_min_loss) {
      global_min_loss    = current_loss;
      global_best_center = particle.center;
      global_best_width  = particle.width;
    }
  }

  for (uint64_t iteration = 0; iteration < max_iterations; ++iteration) {
    float current_inertia =
        initial_inertia - (initial_inertia - final_inertia) * static_cast<float>(iteration) / max_iterations;

    for (auto &particle : particle_swarm) {
      float random_r1 = probability_distribution(random_engine);
      float random_r2 = probability_distribution(random_engine);

      particle.velocity_center = current_inertia * particle.velocity_center +
                                 c1 * random_r1 * (particle.best_center - particle.center) +
                                 c2 * random_r2 * (global_best_center - particle.center);

      particle.velocity_width = current_inertia * particle.velocity_width +
                                c1 * random_r1 * (particle.best_width - particle.width) +
                                c2 * random_r2 * (global_best_width - particle.width);

      particle.center += particle.velocity_center;
      particle.width += particle.velocity_width;

      if (particle.width <= std::numeric_limits<float>::epsilon()) {
        particle.width = std::numeric_limits<float>::epsilon();
      }

      float current_lower = particle.center - particle.width * 0.5f;
      float current_error = loss(div, current_lower, particle.width / div, data_start, data_end);

      if (current_error < particle.min_loss_value) {
        particle.min_loss_value = current_error;
        particle.best_center    = particle.center;
        particle.best_width     = particle.width;
      }

      if (current_error < global_min_loss) {
        global_min_loss    = current_error;
        global_best_center = particle.center;
        global_best_width  = particle.width;
      }
    }
  }

  const float optimized_lower = global_best_center - global_best_width * 0.5f;
  const float optimized_upper = global_best_center + global_best_width * 0.5f;
  return {optimized_lower, optimized_upper};
}