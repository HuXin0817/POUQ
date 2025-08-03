#pragma once

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

namespace pouq {

static constexpr int max_iter = 128;
static constexpr int particle_count = 64;
static constexpr float scale_factor = 0.1f;
static constexpr float init_inertia = 0.9f;
static constexpr float final_inertia = 0.4f;
static constexpr float init_c1 = 2.5f;
static constexpr float final_c1 = 0.5f;
static constexpr float init_c2 = 0.5f;
static constexpr float final_c2 = 2.5f;

struct Particle {
  float center;
  float width;
  float v_center;
  float v_width;
  float best_center;
  float best_width;
  float min_loss;

  Particle(float c_val, float w_val, float vc_val, float vw_val)
      : center(c_val),
        width(w_val),
        v_center(vc_val),
        v_width(vw_val),
        best_center(c_val),
        best_width(w_val),
        min_loss(std::numeric_limits<float>::max()) {
  }
};

static float
loss(float div,
     float cluster_lower_bound,
     float step_size_value,
     const std::vector<std::pair<float, int>>::const_iterator& data_begin,
     const std::vector<std::pair<float, int>>::const_iterator& data_end) {
  step_size_value = std::max(step_size_value, 1e-8f);
  float total_loss = 0.0f;

  for (auto it = data_begin; it != data_end; ++it) {
    const auto& [data_value, point_count] = *it;
    const float real_quantized_code = (data_value - cluster_lower_bound) / step_size_value;
    float quantized_code = 0.0f;

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

std::pair<float, float>
optimize(float div,
         float init_lower_bound,
         float init_upper_bound,
         const std::vector<std::pair<float, int>>::const_iterator& data_begin,
         const std::vector<std::pair<float, int>>::const_iterator& data_end) {
  const float init_range_width = init_upper_bound - init_lower_bound;
  const float init_range_center = (init_lower_bound + init_upper_bound) * 0.5f;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution v_dis(-init_range_width * 0.1f, init_range_width * 0.1f);
  std::uniform_real_distribution p_dis(0.0f, 1.0f);
  std::uniform_real_distribution center_dis(init_range_center - init_range_width * scale_factor,
                                            init_range_center + init_range_width * scale_factor);
  std::uniform_real_distribution width_dis(init_range_width * (1.0f - scale_factor),
                                           init_range_width * (1.0f + scale_factor));

  std::vector<Particle> swarm;
  swarm.reserve(particle_count);
  for (int i = 0; i < particle_count; i++) {
    swarm.emplace_back(center_dis(gen), width_dis(gen), v_dis(gen), v_dis(gen));
  }

  float global_best_center = init_range_center;
  float global_best_width = init_range_width;
  float global_min_loss = loss(div, init_lower_bound, init_range_width / div, data_begin, data_end);

  for (auto& particle : swarm) {
    const float curr_lower_bound = particle.center - particle.width * 0.5f;
    const float curr_loss = loss(div, curr_lower_bound, particle.width / div, data_begin, data_end);

    particle.min_loss = curr_loss;
    if (curr_loss < global_min_loss) {
      global_min_loss = curr_loss;
      global_best_center = particle.center;
      global_best_width = particle.width;
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

      particle.v_center = inertia * particle.v_center +
                          c1 * r1 * (particle.best_center - particle.center) +
                          c2 * r2 * (global_best_center - particle.center);

      particle.v_width = inertia * particle.v_width +
                         c1 * r1 * (particle.best_width - particle.width) +
                         c2 * r2 * (global_best_width - particle.width);

      particle.center += particle.v_center;
      particle.width += particle.v_width;

      if (particle.width <= std::numeric_limits<float>::epsilon()) {
        particle.width = std::numeric_limits<float>::epsilon();
      }

      const float curr_lower = particle.center - particle.width * 0.5f;
      const float curr_loss = loss(div, curr_lower, particle.width / div, data_begin, data_end);

      if (curr_loss < particle.min_loss) {
        particle.min_loss = curr_loss;
        particle.best_center = particle.center;
        particle.best_width = particle.width;
      }

      if (curr_loss < global_min_loss) {
        global_min_loss = curr_loss;
        global_best_center = particle.center;
        global_best_width = particle.width;
      }
    }
  }

  const float opt_lower = global_best_center - global_best_width * 0.5f;
  const float opt_upper = global_best_center + global_best_width * 0.5f;
  return {opt_lower, opt_upper};
}

}  // namespace pouq
