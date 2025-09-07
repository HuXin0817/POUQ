#include "optimize.h"

#include <cassert>
#include <random>

struct Particle {
  float lower;
  float step;
  float v_lower;
  float v_step;
  float best_lower;
  float best_step;
  float min_loss;
};

std::pair<float, float>
optimize(float div,
         float init_lower,
         float init_upper,
         const float* data_map,
         const int* freq_map,
         int size,
         const Parameter& parameter,
         bool do_count_freq) {
  assert(div > 0.0f);
  assert(init_lower <= init_upper);
  assert(size > 0);
  assert(data_map != nullptr);
  if (do_count_freq) {
    assert(freq_map != nullptr);
  }
  assert(parameter.max_iter >= 0);
  assert(parameter.particle_count >= 0);
  assert(parameter.scale_factor >= 0.0f);
  assert(parameter.init_inertia >= 0.0f);
  assert(parameter.final_inertia >= 0.0f);
  assert(parameter.c1 >= 0.0f);
  assert(parameter.c2 >= 0.0f);

  float init_range_width = init_upper - init_lower;
  float init_step = init_range_width / div;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution v_dis(-init_range_width * 0.1f, init_range_width * 0.1f);
  std::uniform_real_distribution p_dis(0.0f, 1.0f);
  std::uniform_real_distribution lower_dis(init_lower - init_range_width * parameter.scale_factor,
                                           init_lower + init_range_width * parameter.scale_factor);
  std::uniform_real_distribution step_dis(init_step * (1.0f - parameter.scale_factor),
                                          init_step * (1.0f + parameter.scale_factor));

  float global_best_lower = init_lower;
  float global_best_step = init_step;
  float global_min_loss = loss(div, init_lower, init_step, data_map, freq_map, size, do_count_freq);

  std::vector<Particle> swarm(parameter.particle_count);
  for (auto& particle : swarm) {
    float lower = lower_dis(gen);
    float step = step_dis(gen);
    float v_lower = v_dis(gen);
    float v_step = v_dis(gen);
    float min_loss = loss(div, lower, step, data_map, freq_map, size, do_count_freq);

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

  for (int iter = 0; iter < parameter.max_iter; ++iter) {
    float x = (float)(iter) / (float)(parameter.max_iter);
    float inertia = parameter.init_inertia - (parameter.init_inertia - parameter.final_inertia) * x;

    for (auto& particle : swarm) {
      float r1 = p_dis(gen);
      float r2 = p_dis(gen);

      particle.v_lower = inertia * particle.v_lower +
                         parameter.c1 * r1 * (particle.best_lower - particle.lower) +
                         parameter.c2 * r2 * (global_best_lower - particle.lower);

      particle.v_step = inertia * particle.v_step +
                        parameter.c1 * r1 * (particle.best_step - particle.step) +
                        parameter.c2 * r2 * (global_best_step - particle.step);

      particle.lower += particle.v_lower;
      particle.step += particle.v_step;
      particle.step = std::max(std::abs(particle.step), FLT_EPSILON);

      float curr_loss =
          loss(div, particle.lower, particle.step, data_map, freq_map, size, do_count_freq);
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
