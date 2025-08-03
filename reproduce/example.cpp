#include "../libpouq/quantizer.hpp"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

static constexpr size_t Dim = 256;
static constexpr size_t N   = Dim * 1000;

int main() {
  std::vector<float> data(N);

  std::random_device                 rd;
  std::mt19937                       gen(rd());
  std::uniform_int_distribution<int> uid(0, 255);

  float l = 255.0f, u = 0.0f;

#pragma omp parallel for reduction(min : l) reduction(max : u)
  for (size_t i = 0; i < N; ++i) {
    data[i] = static_cast<float>(uid(gen));
    l       = std::min(l, data[i]);
    u       = std::max(u, data[i]);
  }

  float step_size = (u - l) / 15.0f;

  float mse_t = 0.0f;
#pragma omp parallel for reduction(+ : mse_t)
  for (size_t i = 0; i < N; ++i) {
    float code = std::round((data[i] - l) / step_size);
    code       = std::clamp(code, 0.0f, 15.0f);
    float q    = code * step_size + l;
    float dif  = q - data[i];
    mse_t += dif * dif;
  }
  std::cout << "Traditional UQ MSE: " << mse_t / N << std::endl;

  pouq::POUQ4bitSIMDQuantizer quantizer(Dim);
  quantizer.train(data.data(), N);

  float mse_p = 0.0f;
#pragma omp parallel for reduction(+ : mse_p)
  for (size_t i = 0; i < N; i += Dim) {
    mse_p += quantizer.l2distance(data.data() + i, i);
  }
  std::cout << "POUQ MSE: " << mse_p / N << std::endl;

  return 0;
}
