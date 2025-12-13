#include "optimize.h"

typedef struct {
  float lower;
  float step;
  float v_lower;
  float v_step;
  float best_lower;
  float best_step;
  float min_loss;
} Particle;

float
loss(float lower,
     float step,
     const float* data_map,
     const int* freq_map,
     int size,
     bool do_count_freq) {
  assert(step >= FLT_EPSILON);
  assert(size > 0);
  assert(data_map != NULL);
  if (do_count_freq) {
    assert(freq_map != NULL);
  }

  float total_loss = 0.0f;

#pragma omp simd
  for (int i = 0; i < size; ++i) {
    float data_value = data_map[i];
    float real_quantized_code = (data_value - lower) / step;
    float quantized_code = 0.0f;

    if (data_value > lower) {
      if (real_quantized_code < 3.0f) {
        quantized_code = roundf(real_quantized_code);
      } else {
        quantized_code = 3.0f;
      }
    }

    float code_loss = real_quantized_code - quantized_code;
    if (do_count_freq) {
      total_loss += code_loss * code_loss * (float)(freq_map[i]);
    } else {
      total_loss += code_loss * code_loss;
    }
  }

  return total_loss * step * step;
}

Bound
optimize(float init_lower,
         float init_upper,
         const float* data_map,
         const int* freq_map,
         int size,
         const Parameter parameter,
         bool do_count_freq) {
  assert(init_lower <= init_upper);
  assert(size > 0);
  assert(data_map != NULL);
  if (do_count_freq) {
    assert(freq_map != NULL);
  }
  assert(parameter.max_iter >= 0);
  assert(parameter.particle_count >= 0);
  assert(parameter.scale_factor >= 0.0f);
  assert(parameter.init_inertia >= 0.0f);
  assert(parameter.final_inertia >= 0.0f);
  assert(parameter.c1 >= 0.0f);
  assert(parameter.c2 >= 0.0f);

  float init_range_width = init_upper - init_lower;
  float init_step = init_range_width / 3.0f;

  float v_min = -init_range_width * 0.1f;
  float v_max = init_range_width * 0.1f;
  float lower_min = init_lower - init_range_width * parameter.scale_factor;
  float lower_max = init_lower + init_range_width * parameter.scale_factor;
  float step_min = init_step * (1.0f - parameter.scale_factor);
  float step_max = init_step * (1.0f + parameter.scale_factor);

  float global_best_lower = init_lower;
  float global_best_step = init_step;
  float global_min_loss = loss(init_lower, init_step, data_map, freq_map, size, do_count_freq);

  Particle* swarm = malloc(parameter.particle_count * sizeof(Particle));

  for (int i = 0; i < parameter.particle_count; i++) {
    float lower = rand_float(lower_min, lower_max);
    float step = rand_float(step_min, step_max);
    float v_lower = rand_float(v_min, v_max);
    float v_step = rand_float(v_min, v_max);
    float min_loss = loss(lower, step, data_map, freq_map, size, do_count_freq);

    swarm[i] = (Particle){
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
      global_best_lower = swarm[i].lower;
      global_best_step = swarm[i].step;
    }
  }

  for (int iter = 0; iter < parameter.max_iter; ++iter) {
    float x = (float)iter / (float)parameter.max_iter;
    float inertia = parameter.init_inertia - (parameter.init_inertia - parameter.final_inertia) * x;

    for (int i = 0; i < parameter.particle_count; i++) {
      float r1 = rand_float(0.0f, 1.0f);
      float r2 = rand_float(0.0f, 1.0f);
      Particle* particle = &swarm[i];

      particle->v_lower = inertia * particle->v_lower +
                          parameter.c1 * r1 * (particle->best_lower - particle->lower) +
                          parameter.c2 * r2 * (global_best_lower - particle->lower);

      particle->v_step = inertia * particle->v_step +
                         parameter.c1 * r1 * (particle->best_step - particle->step) +
                         parameter.c2 * r2 * (global_best_step - particle->step);

      particle->lower += particle->v_lower;
      particle->step += particle->v_step;
      particle->step = fmaxf(fabsf(particle->step), FLT_EPSILON);

      float curr_loss =
          loss(particle->lower, particle->step, data_map, freq_map, size, do_count_freq);
      if (curr_loss < particle->min_loss) {
        particle->min_loss = curr_loss;
        particle->best_lower = particle->lower;
        particle->best_step = particle->step;

        if (curr_loss < global_min_loss) {
          global_min_loss = curr_loss;
          global_best_lower = particle->lower;
          global_best_step = particle->step;
        }
      }
    }
  }

  free(swarm);

  Bound result;
  result.lower = global_best_lower;
  result.upper = global_best_lower + global_best_step * 3.0f;
  return result;
}
