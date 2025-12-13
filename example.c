#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libpouq/distance.h"
#include "libpouq/train.h"
#include "libpouq/util.h"

const int Dim = 256;

int
main() {
  int N = 1000;
  N *= Dim;

  float* data = malloc(N * sizeof(float));

  srand(time(NULL));

  printf("Data:     [");
  for (int i = 0; i < N; ++i) {
    data[i] = rand_float(0.0f, 1.0f);
    if (i < 5) {
      printf("%f, ", data[i]);
    }
  }
  printf("...]\n");

  float lower[Dim], upper[Dim];
  for (int i = 0; i < Dim; i++) {
    lower[i] = data[i];
    upper[i] = data[i];
    for (int j = 0; j < (N / Dim); j++) {
      lower[i] = min(lower[i], data[j * Dim + i]);
      upper[i] = max(upper[i], data[j * Dim + i]);
    }
  }

  float step_size[Dim];
  for (int i = 0; i < Dim; i++) {
    step_size[i] = (upper[i] - lower[i]) / 15.0f;
  }

  float baseline_mse = 0.0f;
  for (int i = 0; i < N; i++) {
    int d = i % Dim;
    int x = (data[i] - lower[d]) / step_size[d] + 0.5f;
    float decode = x * step_size[d] + lower[d];

    baseline_mse += (data[i] - decode) * (data[i] - decode);
  }

  printf("Baseline: %.15f\n", baseline_mse / N);

  Parameter param = {
      .max_iter = 100,
      .particle_count = 50,
      .scale_factor = 0.1f,
      .init_inertia = 0.9f,
      .final_inertia = 0.4f,
      .c1 = 1.5f,
      .c2 = 1.5f,
  };

  Result result = train(Dim, data, N, param);
  if (!result.code || !result.rec_para) {
    free(data);
    printf("train error\n");
    return 1;
  }

  float mse = 0.0f;
#pragma omp parallel for reduction(+ : mse)
  for (int i = 0; i < N; i += Dim) {
    mse += distance(Dim, result.code, result.rec_para, data + i, i);
  }
  printf("Ours:     %.15f\n", mse / N);

  free(result.code);
  free(result.rec_para);
  free(data);
  return 0;
}
