#include <iostream>

#include "POUQ/libpouq.hpp"

static constexpr int Dim = 256;
static constexpr int N = Dim * 1e5;

int
main() {
  std::vector<float> data(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution dis(0.0f, 255.0f);

  float l = 255.0f, u = 0.0f;

#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    data[i] = dis(gen);
  }

  pouq::Quantizer quantizer(Dim);
  quantizer.train(data.data(), N);

  float mse_p = 0.0f;
#pragma omp parallel for reduction(+ : mse_p)
  for (int i = 0; i < N; i += Dim) {
    mse_p += quantizer.distance(data.data() + i, i);
  }
  std::cout << mse_p / N << std::endl;

  return 0;
}
