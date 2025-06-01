#pragma once

#include <cstdint>

namespace pouq::bitmap {

inline void set(uint8_t *data, size_t index, size_t n, size_t bit_size) {
  if (bit_size == 0) {
    return;
  }

  n &= (1 << bit_size) - 1;
  const size_t pos = index * bit_size;
  for (size_t bit = 0; bit < bit_size; ++bit) {
    const size_t i      = (pos + bit) / 8;
    const size_t offset = (pos + bit) % 8;
    if (n & 1 << bit) {
      data[i] |= 1 << offset;
    } else {
      data[i] &= ~(1 << offset);
    }
  }
}

inline size_t get(const uint8_t *data, size_t index, size_t bit_size) {
  if (bit_size == 0) {
    return 0;
  }

  const size_t pos    = index * bit_size;
  size_t       result = 0;
  for (size_t bit = 0; bit < bit_size; ++bit) {
    const size_t i      = (pos + bit) / 8;
    const size_t offset = (pos + bit) % 8;
    if (data[i] & 1 << offset) {
      result |= 1 << bit;
    }
  }

  return result;
}

}  // namespace pouq::bitmap
