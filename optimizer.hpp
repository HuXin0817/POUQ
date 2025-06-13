#pragma once

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

namespace pouq {

static constexpr size_t MAX_ITERATIONS           = 128;
static constexpr size_t PARTICLE_COUNT           = 64;
static constexpr float  GRID_SCALE_FACTOR        = 0.1f;
static constexpr float  INITIAL_INERTIA          = 0.9f;
static constexpr float  FINAL_INERTIA            = 0.4f;
static constexpr float  PERSONAL_LEARNING_FACTOR = 1.8f;
static constexpr float  SOCIAL_LEARNING_FACTOR   = 1.8f;

struct Particle {
  float position_center;
  float position_width;
  float velocity_center;
  float velocity_width;
  float best_position_center;
  float best_position_width;
  float best_loss;

  Particle(const float center_val,
      const float      width_val,
      const float      velocity_center_val,
      const float      velocity_width_val)
      : position_center(center_val),
        position_width(width_val),
        velocity_center(velocity_center_val),
        velocity_width(velocity_width_val),
        best_position_center(center_val),
        best_position_width(width_val),
        best_loss(std::numeric_limits<float>::max()) {}
};

inline float calculate_quantization_loss(const float             division_count,
    float                                                        cluster_lower_bound,
    float                                                        step_size,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_end) {
  step_size        = std::max(step_size, 1e-8f);
  float total_loss = 0.0f;

  for (auto it = data_begin; it != data_end; ++it) {
    const auto &[data_value, point_count] = *it;
    const float real_quantized_code       = (data_value - cluster_lower_bound) / step_size;
    float       quantized_code            = 0.0f;

    if (data_value > cluster_lower_bound) {
      quantized_code = std::round(real_quantized_code);
      if (quantized_code > division_count) {
        quantized_code = division_count;
      }
    }

    const float quantization_error = real_quantized_code - quantized_code;
    total_loss += quantization_error * quantization_error * static_cast<float>(point_count);
  }

  return total_loss * step_size * step_size;
}

inline std::pair<float, float> optimize_quantization_range(float division_count,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_start,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_end) {
  auto        initial_lower_bound  = data_start->first;
  auto        initial_upper_bound  = (data_end - 1)->first;
  const float initial_range_width  = initial_upper_bound - initial_lower_bound;
  const float initial_range_center = (initial_lower_bound + initial_upper_bound) * 0.5f;

  std::random_device             random_device;
  std::mt19937                   random_generator(random_device());
  std::uniform_real_distribution velocity_distribution(-initial_range_width * 0.1f, initial_range_width * 0.1f);
  std::uniform_real_distribution random_factor_distribution(0.0f, 1.0f);

  const float                    expanded_lower_bound = initial_lower_bound - GRID_SCALE_FACTOR * initial_range_width;
  const float                    expanded_upper_bound = initial_upper_bound + GRID_SCALE_FACTOR * initial_range_width;
  std::uniform_real_distribution center_distribution(expanded_lower_bound, expanded_upper_bound);
  std::uniform_real_distribution width_distribution(initial_range_width * 0.1f, initial_range_width * 1.2f);

  std::vector<Particle> particle_swarm;
  particle_swarm.reserve(PARTICLE_COUNT);

  for (size_t i = 0; i < PARTICLE_COUNT; i++) {
    const float particle_center = center_distribution(random_generator);
    const float particle_width  = width_distribution(random_generator);

    particle_swarm.emplace_back(particle_center,
        particle_width,
        velocity_distribution(random_generator),
        velocity_distribution(random_generator));
  }

  float global_best_center = initial_range_center;
  float global_best_width  = initial_range_width;
  float global_min_loss    = calculate_quantization_loss(
      division_count, initial_lower_bound, initial_range_width / division_count, data_start, data_end);

  for (auto &particle : particle_swarm) {
    const float current_lower_bound = particle.position_center - particle.position_width * 0.5f;
    const float current_loss        = calculate_quantization_loss(
        division_count, current_lower_bound, particle.position_width / division_count, data_start, data_end);

    particle.best_loss = current_loss;
    if (current_loss < global_min_loss) {
      global_min_loss    = current_loss;
      global_best_center = particle.position_center;
      global_best_width  = particle.position_width;
    }
  }

  for (size_t iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
    const float inertia_weight =
        INITIAL_INERTIA - (INITIAL_INERTIA - FINAL_INERTIA) * static_cast<float>(iteration) / MAX_ITERATIONS;

    for (auto &particle : particle_swarm) {
      const float cognitive_random_factor = random_factor_distribution(random_generator);
      const float social_random_factor    = random_factor_distribution(random_generator);

      particle.velocity_center =
          inertia_weight * particle.velocity_center +
          PERSONAL_LEARNING_FACTOR * cognitive_random_factor *
              (particle.best_position_center - particle.position_center) +
          SOCIAL_LEARNING_FACTOR * social_random_factor * (global_best_center - particle.position_center);

      particle.velocity_width =
          inertia_weight * particle.velocity_width +
          PERSONAL_LEARNING_FACTOR * cognitive_random_factor *
              (particle.best_position_width - particle.position_width) +
          SOCIAL_LEARNING_FACTOR * social_random_factor * (global_best_width - particle.position_width);

      particle.position_center += particle.velocity_center;
      particle.position_width += particle.velocity_width;

      if (particle.position_width <= std::numeric_limits<float>::epsilon()) {
        particle.position_width = std::numeric_limits<float>::epsilon();
      }

      const float current_lower_bound = particle.position_center - particle.position_width * 0.5f;
      const float current_loss        = calculate_quantization_loss(
          division_count, current_lower_bound, particle.position_width / division_count, data_start, data_end);

      if (current_loss < particle.best_loss) {
        particle.best_loss            = current_loss;
        particle.best_position_center = particle.position_center;
        particle.best_position_width  = particle.position_width;
      }

      if (current_loss < global_min_loss) {
        global_min_loss    = current_loss;
        global_best_center = particle.position_center;
        global_best_width  = particle.position_width;
      }
    }
  }

  const float optimal_lower_bound = global_best_center - global_best_width * 0.5f;
  const float optimal_upper_bound = global_best_center + global_best_width * 0.5f;
  return {optimal_lower_bound, optimal_upper_bound};
}

}  // namespace pouq
