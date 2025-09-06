#include <iostream>

#include "libpouq/train.h"

static constexpr int Dim = 256;
static constexpr int N = Dim * 1e5;

int
main() {
  std::vector<float> data(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution dis(0.0f, 255.0f);

#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    data[i] = dis(gen);
  }

  auto res = pouq::train(Dim, data.data(), N);

  float mse = 0.0f;
#pragma omp parallel for reduction(+ : mse)
  for (int i = 0; i < N; i += Dim) {
    mse += pouq::simd::distance(Dim, res.first, res.second, data.data() + i, i);
  }
  std::cout << mse / N << std::endl;

  pouq::_free(res.first);
  pouq::_free(res.second);
  res.first = nullptr;
  res.second = nullptr;
  return 0;
}
