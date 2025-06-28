#pragma once

#include <cstdint>

namespace pouq::bitmap {

inline void set(uint8_t *data, size_t index, size_t n, size_t bit_size) {
  if (bit_size == 0) {
    return;
  }

  n &= (1 << bit_size) - 1;
  size_t pos = index * bit_size;

  for (size_t bit = 0; bit < bit_size; ++bit) {
    size_t byte_idx   = (pos + bit) / 8;
    size_t bit_offset = (pos + bit) % 8;

    if (n & 1 << bit) {
      data[byte_idx] |= 1 << bit_offset;
    } else {
      data[byte_idx] &= ~(1 << bit_offset);
    }
  }
}

inline size_t get(const uint8_t *data, size_t index, size_t bit_size) {
  if (bit_size == 0) {
    return 0;
  }

  size_t pos    = index * bit_size;
  size_t result = 0;

  for (size_t bit = 0; bit < bit_size; ++bit) {
    size_t byte_idx   = (pos + bit) / 8;
    size_t bit_offset = (pos + bit) % 8;

    if (data[byte_idx] & (1 << bit_offset)) {
      result |= 1 << bit;
    }
  }

  return result;
}

}  // namespace pouq::bitmap
