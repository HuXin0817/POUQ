#pragma once

#include <omp.h>

template <typename D1, typename D2, typename T>
float compute_mse(const D1 &d1, const D2 &d2, T size) {
  float mse = 0;
#pragma omp parallel for reduction(+ : mse)
  for (T i = 0; i < size; ++i) {
    const float dif = d1[i] - d2[i];
    mse += dif * dif;
  }
  return mse / static_cast<float>(size);
}
