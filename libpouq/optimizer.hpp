#pragma once

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

namespace pouq {

static constexpr int max_iter = 100;
static constexpr int particle_count = 50;
static constexpr float scale_factor = 0.1f;
static constexpr float init_inertia = 0.9f;
static constexpr float final_inertia = 0.4f;
static constexpr float init_c1 = 2.5f;
static constexpr float final_c1 = 0.5f;
static constexpr float init_c2 = 0.5f;
static constexpr float final_c2 = 2.5f;

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

std::pair<float, float>
optimize(float div,
         float init_lower,
         float init_upper,
         const std::vector<std::pair<float, int>>::const_iterator& data_begin,
         const std::vector<std::pair<float, int>>::const_iterator& data_end) {
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

}  // namespace pouq
