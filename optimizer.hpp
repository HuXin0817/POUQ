#pragma once

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "simd.hpp"

namespace pouq {

static constexpr size_t MAX_ITERATIONS    = 128;
static constexpr size_t PARTICLE_COUNT    = 50;
static constexpr float  GRID_SCALE_FACTOR = 0.1f;
static constexpr float  INITIAL_INERTIA   = 0.9f;
static constexpr float  FINAL_INERTIA     = 0.4f;
static constexpr float  C1                = 1.8f;
static constexpr float  C2                = 1.8f;

struct Particle {
  float position_lower;
  float position_upper;
  float velocity_lower;
  float velocity_upper;
  float best_position_lower;
  float best_position_upper;
  float best_loss;

  Particle(const float lower_val, const float upper_val, const float velocity_lower_val, const float velocity_upper_val)
      : position_lower(lower_val),
        position_upper(upper_val),
        velocity_lower(velocity_lower_val),
        velocity_upper(velocity_upper_val),
        best_position_lower(lower_val),
        best_position_upper(upper_val),
        best_loss(std::numeric_limits<float>::max()) {}
};

inline std::pair<float, float> optimize_quantization_range(float division_count,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_start,
    const std::vector<std::pair<float, size_t>>::const_iterator &data_end) {
  const auto  initial_lower_bound = data_start->first;
  const auto  initial_upper_bound = (data_end - 1)->first;
  const float initial_range_width = initial_upper_bound - initial_lower_bound;

  std::random_device             random_device;
  std::mt19937                   random_generator(random_device());
  std::uniform_real_distribution velocity_distribution(-initial_range_width * 0.1f, initial_range_width * 0.1f);
  std::uniform_real_distribution random_factor_distribution(0.0f, 1.0f);
  std::uniform_real_distribution lower_distribution(initial_lower_bound - GRID_SCALE_FACTOR * initial_range_width,
      initial_lower_bound + GRID_SCALE_FACTOR * initial_range_width);
  std::uniform_real_distribution upper_distribution(initial_upper_bound - GRID_SCALE_FACTOR * initial_range_width,
      initial_upper_bound + GRID_SCALE_FACTOR * initial_range_width);

  std::vector<Particle> particle_swarm;
  particle_swarm.reserve(PARTICLE_COUNT);
  for (size_t i = 0; i < PARTICLE_COUNT; i++) {
    particle_swarm.emplace_back(lower_distribution(random_generator),
        upper_distribution(random_generator),
        velocity_distribution(random_generator),
        velocity_distribution(random_generator));
  }

  float global_best_lower = initial_lower_bound;
  float global_best_upper = initial_upper_bound;
  float global_min_loss   = calculate_quantization_loss(
      division_count, initial_lower_bound, initial_range_width / division_count, data_start, data_end);

  for (auto &particle : particle_swarm) {
    const float current_width = particle.position_upper - particle.position_lower;
    const float current_loss  = calculate_quantization_loss(
        division_count, particle.position_lower, current_width / division_count, data_start, data_end);

    particle.best_loss = current_loss;
    if (current_loss < global_min_loss) {
      global_min_loss   = current_loss;
      global_best_lower = particle.position_lower;
      global_best_upper = particle.position_upper;
    }
  }

  for (size_t iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
    const float inertia_weight =
        INITIAL_INERTIA - (INITIAL_INERTIA - FINAL_INERTIA) * static_cast<float>(iteration) / MAX_ITERATIONS;

    for (auto &particle : particle_swarm) {
      const float cognitive_random_factor = random_factor_distribution(random_generator);
      const float social_random_factor    = random_factor_distribution(random_generator);

      particle.velocity_lower =
          inertia_weight * particle.velocity_lower +
          C1 * cognitive_random_factor * (particle.best_position_lower - particle.position_lower) +
          C2 * social_random_factor * (global_best_lower - particle.position_lower);

      particle.velocity_upper =
          inertia_weight * particle.velocity_upper +
          C1 * cognitive_random_factor * (particle.best_position_upper - particle.position_upper) +
          C2 * social_random_factor * (global_best_upper - particle.position_upper);

      particle.position_lower += particle.velocity_lower;
      particle.position_upper += particle.velocity_upper;

      if (particle.position_lower > particle.position_upper) {
        std::swap(particle.position_lower, particle.position_upper);
        std::swap(particle.velocity_lower, particle.velocity_upper);
      }

      constexpr float min_width = std::numeric_limits<float>::epsilon();
      if (particle.position_upper - particle.position_lower < min_width) {
        const float center      = (particle.position_lower + particle.position_upper) * 0.5f;
        particle.position_lower = center - min_width * 0.5f;
        particle.position_upper = center + min_width * 0.5f;
      }

      const float current_width = particle.position_upper - particle.position_lower;
      const float current_loss  = calculate_quantization_loss(
          division_count, particle.position_lower, current_width / division_count, data_start, data_end);

      if (current_loss < particle.best_loss) {
        particle.best_loss           = current_loss;
        particle.best_position_lower = particle.position_lower;
        particle.best_position_upper = particle.position_upper;
      }

      if (current_loss < global_min_loss) {
        global_min_loss   = current_loss;
        global_best_lower = particle.position_lower;
        global_best_upper = particle.position_upper;
      }
    }
  }

  return {global_best_lower, global_best_upper};
}

}  // namespace pouq
