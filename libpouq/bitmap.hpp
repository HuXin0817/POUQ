#pragma once

#include <cstdint>

namespace pouq::bitmap {

inline void set(uint8_t *data, size_t index, size_t n) {
  n &= 3;
  const size_t i      = index * 2 / 8;
  const size_t offset = index * 2 % 8;
  data[i] &= ~(3 << offset);
  data[i] |= (n << offset);
}

inline size_t get(const uint8_t *data, size_t index) {
  const size_t i      = (index * 2) / 8;
  const size_t offset = (index * 2) % 8;
  return (data[i] >> offset) & 3;
}

}  // namespace pouq::bitmap