#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libpouq/distance.h"
#include "libpouq/optimize.h"
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

  SQ4Result sq4_result = train_sq4(Dim, data, N);
  float mse = 0.0f;
#pragma omp parallel for reduction(+ : mse)
  for (int i = 0; i < N; i += Dim) {
    mse += distance_sq4(Dim, sq4_result.code, sq4_result.rec_para, data + i, i);
  }
  printf("Baseline: %.15f\n", mse / N);

  free(sq4_result.code);
  free(sq4_result.rec_para);

  Result result = train(Dim, data, N);
  mse = 0.0f;
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
