#include "../libpouq/quantizer.hpp"
#include <iostream>

static constexpr size_t N = 256 * 1e5;

int main(int argc, char *argv[]) {
  std::vector<float> data(N);

  std::random_device             rd;
  std::mt19937                   gen(rd());
  std::uniform_real_distribution urd(0.0f, 1.0f);

  float l = 0.0f, u = 1.0f;

#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    data[i] = urd(gen);
    l       = std::min(l, data[i]);
    u       = std::max(u, data[i]);
  }

  float step_size = (u - l) / 15;

  float mse_t = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float code = std::round((data[i] - l) / step_size);
    float q    = code * step_size + l;
    float dif  = q - data[i];
    mse_t += dif * dif;
  }

  std::cout << mse_t << std::endl;

  pouq::POUQ4bitSIMDQuantizer quantizer(256);
  quantizer.train(data.data(), N);
  std::cout << quantizer.l2distance(data.data(), N) << std::endl;
  return 0;
}
