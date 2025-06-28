#pragma once

#include <cstdint>
#include <utility>

namespace pouq::bitmap {

inline void set4bit(uint8_t *data, size_t i, size_t v1, size_t v2) { data[i] = (v1 & 0xF) * 16 + (v2 & 0xF); }

inline void set2bit(uint8_t *data, size_t i, size_t v1, size_t v2) {
  size_t v        = (v1 & 0x3) * 4 + (v2 & 0x3);
  size_t byte_idx = i / 2;
  size_t offset   = (i % 2) * 4;
  data[byte_idx]  = (data[byte_idx] & ~(0xF << offset)) | (v << offset);
}

inline std::pair<size_t, size_t> get4bit(const uint8_t *data, size_t i) {
  auto v = data[i];
  return {v / 16, v % 16};
}

inline std::pair<size_t, size_t> get2bit(const uint8_t *data, size_t i) {
  size_t byte_idx = i / 2;
  size_t offset   = (i % 2) * 4;
  size_t v        = (data[byte_idx] >> offset) & 0xF;
  return {v / 4, v % 4};
}

}  // namespace pouq::bitmap
