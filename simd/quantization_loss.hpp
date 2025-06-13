#include <vector>

namespace pouq::simd {

inline float quantization_loss_simd(const float                  d,
    float                                                        l,
    float                                                        s,
    const std::vector<std::pair<float, size_t>>::const_iterator &begin,
    const std::vector<std::pair<float, size_t>>::const_iterator &end) {
  s          = std::max(s, 1e-8f);
  float loss = 0.0f;

  for (auto it = begin; it != end; ++it) {
    const auto &[v, c] = *it;
    const float rc     = (v - l) / s;
    float       qc     = 0.0f;
    if (v > l) {
      qc = std::round(rc);
      if (qc > d) {
        qc = d;
      }
    }
    const float err = rc - qc;
    loss += err * err * static_cast<float>(c);
  }

  return loss * s * s;
}

}  // namespace pouq::simd