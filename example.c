#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libpouq/distance.h"
#include "libpouq/train.h"

int
get_padded_dim(int dim) {
  if (dim % 8 == 0) {
    return dim;
  }
  return (dim / 8 + 1) * 8;
}

float*
read_fvecs_padded(const char* filename, int* orig_dim, int* new_dim, int* num_vec) {
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    perror("Failed to open file");
    return NULL;
  }

  int32_t first_dim;
  if (fread(&first_dim, sizeof(int32_t), 1, fp) != 1) {
    perror("Failed to read dimension");
    fclose(fp);
    return NULL;
  }
  *orig_dim = first_dim;

  size_t vec_byte_size = sizeof(int32_t) + *orig_dim * sizeof(float);

  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  *num_vec = file_size / vec_byte_size;
  fseek(fp, 0, SEEK_SET);

  *new_dim = get_padded_dim(*orig_dim);

  float* padded_vecs = (float*)malloc(*num_vec * *new_dim * sizeof(float));
  if (!padded_vecs) {
    perror("Failed to allocate memory");
    fclose(fp);
    return NULL;
  }

  for (int i = 0; i < *num_vec; i++) {
    int32_t current_dim;
    if (fread(&current_dim, sizeof(int32_t), 1, fp) != 1) {
      perror("Failed to read dimension");
      free(padded_vecs);
      fclose(fp);
      return NULL;
    }
    if (current_dim != *orig_dim) {
      fprintf(stderr, "Vector dimension mismatch (vector %d)\n", i);
      free(padded_vecs);
      fclose(fp);
      return NULL;
    }

    float* dest = padded_vecs + i * *new_dim;

    if (fread(dest, sizeof(float), *orig_dim, fp) != *orig_dim) {
      perror("Failed to read vector data");
      free(padded_vecs);
      fclose(fp);
      return NULL;
    }

    if (*new_dim > *orig_dim) {
      memset(dest + *orig_dim, 0, (*new_dim - *orig_dim) * sizeof(float));
    }
  }

  fclose(fp);
  return padded_vecs;
}

float
baseline(const float* data, int size) {
  float minv = data[0];
  float maxv = data[0];

  for (int i = 1; i < size; i++) {
    minv = min(minv, data[i]);
    maxv = max(maxv, data[i]);
  }

  float step_size = (maxv - minv) / 15.0f;
  float ret = 0.0f;
  for (int i = 0; i < size; i++) {
    int v = (data[i] - minv) / step_size + 0.5f;
    float dif = (step_size * v) + minv - data[i];
    ret += dif * dif;
  }

  return ret / size;
}

int
main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
    return 1;
  }

  srand((unsigned int)time(NULL));
  const char* filename = argv[1];

  int real_dim, dim = 0, size = 0;
  float* data = read_fvecs_padded(filename, &real_dim, &dim, &size);
  size *= dim;

  printf("Baseline: %f\n", baseline(data, size));

  Parameter param = {
      .max_iter = 100,
      .particle_count = 50,
      .scale_factor = 0.1f,
      .init_inertia = 0.9f,
      .final_inertia = 0.4f,
      .c1 = 1.5f,
      .c2 = 1.5f,
  };

  Result result = train(dim, data, size, param);

  float mse = 0.0f;
#pragma omp parallel for reduction(+ : mse)
  for (int i = 0; i < size; i += dim) {
    mse += distance(dim, result.code, result.rec_para, data + i, i);
  }
  printf("Ours: %f\n", mse / size);

  free(result.code);
  free(result.rec_para);
  free(data);
  return 0;
}
