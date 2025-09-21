#include "optimize.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "def.h"

typedef struct {
  float lower;
  float step;
  float v_lower;
  float v_step;
  float best_lower;
  float best_step;
  float min_loss;
} Particle;

static float
rand_float(float min, float max) {
  return min + (float)rand() / (float)RAND_MAX * (max - min);
}

float
loss(float div,
     float lower,
     float step,
     const float* data_map,
     const int* freq_map,
     int size,
     int do_count_freq) {
  assert(div > 0.0f);
  assert(step >= FLT_EPSILON);
  assert(size > 0);
  assert(data_map != NULL);
  if (do_count_freq) {
    assert(freq_map != NULL);
  }

  __m256 lower_vec = _mm256_set1_ps(lower);
  __m256 step_vec = _mm256_set1_ps(step);
  __m256 div_vec = _mm256_set1_ps(div);
  __m256 zero_vec = _mm256_setzero_ps();
  __m256 total_loss_vec = _mm256_setzero_ps();

  int simd_size = size - size % 8;
  for (int i = 0; i < simd_size; i += 8) {
    __m256 data_vec = _mm256_loadu_ps(&data_map[i]);
    __m256 real_quantized_code = _mm256_div_ps(_mm256_sub_ps(data_vec, lower_vec), step_vec);
    __m256 quantized_code = zero_vec;
    __m256 greater_mask = _mm256_cmp_ps(data_vec, lower_vec, _CMP_GT_OS);
    __m256 rounded_code =
        _mm256_round_ps(real_quantized_code, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    quantized_code = _mm256_blendv_ps(zero_vec, rounded_code, greater_mask);
    __m256 clamped_code = _mm256_min_ps(quantized_code, div_vec);
    quantized_code = _mm256_blendv_ps(zero_vec, clamped_code, greater_mask);
    __m256 code_loss = _mm256_sub_ps(real_quantized_code, quantized_code);
    __m256 loss_squared = _mm256_mul_ps(code_loss, code_loss);
    if (do_count_freq) {
      __m128i freq_low = _mm_loadu_si128((const __m128i*)(&freq_map[i]));
      __m128i freq_high = _mm_loadu_si128((const __m128i*)(&freq_map[i + 4]));
      __m256 freq_vec = _mm256_cvtepi32_ps(
          _mm256_inserti128_si256(_mm256_castsi128_si256(freq_low), freq_high, 1));
      loss_squared = _mm256_mul_ps(loss_squared, freq_vec);
    }
    total_loss_vec = _mm256_add_ps(total_loss_vec, loss_squared);
  }

  float total_loss = 0.0f;
  for (int i = simd_size; i < size; ++i) {
    float data_value = data_map[i];
    float real_quantized_code = (data_value - lower) / step;
    float quantized_code = 0.0f;

    if (data_value > lower) {
      quantized_code = roundf(real_quantized_code);
      if (quantized_code > div) {
        quantized_code = div;
      }
    }

    float code_loss = real_quantized_code - quantized_code;
    if (do_count_freq) {
      total_loss += code_loss * code_loss * (float)(freq_map[i]);
    } else {
      total_loss += code_loss * code_loss;
    }
  }

  __m128 sum_low = _mm256_castps256_ps128(total_loss_vec);
  __m128 sum_high = _mm256_extractf128_ps(total_loss_vec, 1);
  __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
  __m128 shuffled = _mm_movehdup_ps(sum_128);
  sum_128 = _mm_add_ps(sum_128, shuffled);
  shuffled = _mm_movehl_ps(shuffled, sum_128);
  sum_128 = _mm_add_ss(sum_128, shuffled);
  total_loss += _mm_cvtss_f32(sum_128);

  return total_loss * step * step;
}

Bound
optimize(float div,
         float init_lower,
         float init_upper,
         const float* data_map,
         const int* freq_map,
         int size,
         const Parameter parameter,
         int do_count_freq) {
  assert(div > 0.0f);
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

  srand((unsigned int)time(NULL));

  float init_range_width = init_upper - init_lower;
  float init_step = init_range_width / div;

  float v_min = -init_range_width * 0.1f;
  float v_max = init_range_width * 0.1f;
  float lower_min = init_lower - init_range_width * parameter.scale_factor;
  float lower_max = init_lower + init_range_width * parameter.scale_factor;
  float step_min = init_step * (1.0f - parameter.scale_factor);
  float step_max = init_step * (1.0f + parameter.scale_factor);

  float global_best_lower = init_lower;
  float global_best_step = init_step;
  float global_min_loss = loss(div, init_lower, init_step, data_map, freq_map, size, do_count_freq);

  Particle* swarm = NULL;
  do_malloc(swarm, Particle, parameter.particle_count);

  for (int i = 0; i < parameter.particle_count; i++) {
    float lower = rand_float(lower_min, lower_max);
    float step = rand_float(step_min, step_max);
    float v_lower = rand_float(v_min, v_max);
    float v_step = rand_float(v_min, v_max);
    float min_loss = loss(div, lower, step, data_map, freq_map, size, do_count_freq);

    swarm[i].lower = lower;
    swarm[i].step = step;
    swarm[i].v_lower = v_lower;
    swarm[i].v_step = v_step;
    swarm[i].best_lower = lower;
    swarm[i].best_step = step;
    swarm[i].min_loss = min_loss;

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
          loss(div, particle->lower, particle->step, data_map, freq_map, size, do_count_freq);
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

cleanup:

  do_free(swarm);

  Bound result;
  result.lower = global_best_lower;
  result.upper = global_best_lower + global_best_step * div;
  return result;
}
