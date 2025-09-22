#include <stdio.h>

#include "libpouq/distance.h"
#include "libpouq/train.h"

const int Dim = 256;
const int N = Dim * 1000;

int
main() {
  float* data = NULL;
  do_malloc(data, float, N);

  srand(time(NULL));

#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    data[i] = rand_float(0.0f, 255.0f);
  }

  Parameter param = {100, 50, 0.1f, 0.9f, 0.4f, 1.5f, 1.5f};
  Result result = train(Dim, data, N, param);
  if (!result.code || !result.rec_para) {
    do_free(data);
    printf("train error\n");
    return 1;
  }

  float mse = 0.0f;
#pragma omp parallel for reduction(+ : mse)
  for (int i = 0; i < N; i += Dim) {
    mse += distance(Dim, result.code, result.rec_para, data + i, i);
  }
  printf("%f\n", mse / N);

cleanup:
  do_free(result.code);
  do_free(result.rec_para);
  do_free(data);
  return 0;
}
