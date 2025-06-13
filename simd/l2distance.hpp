namespace pouq::simd {

inline float l2distance_simd(const float *data,
    size_t                                index,
    size_t                                dim,
    const std::pair<float, float>        *code,
    const uint8_t                        *encode) {
  float dis = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    const uint8_t c   = encode[index + i];
    const auto [l, s] = code[((c & 0xF) + i * 16)];
    const float v     = l + s * (c >> 4 & 0xF);
    const float diff  = data[i] - v;
    dis += diff * diff;
  }
  return dis;
}

}  // namespace pouq::simd