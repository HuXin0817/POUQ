#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libpouq/distance.h"
#include "libpouq/train.h"

const int Dim = 256;

int
main(int argc, char* argv[]) {
  int N = 1000;
  if (argc > 1) {
    N = max(N, atoi(argv[1]));
  }

  N *= Dim;

  float* data = NULL;
  do_malloc(data, N);

  srand(time(NULL));

#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    data[i] = rand_float(0.0f, 255.0f);
  }

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
  printf("%f\n", mse / N);

  free(result.code);
  free(result.rec_para);
  free(data);
  return 0;
}
