#pragma once

#include <cstdint>

namespace pouq {

namespace bitmap {

void set(uint8_t *data, uint64_t index, uint64_t n, uint64_t bit_size) {
  if (bit_size == 0) {
    return;
  }

  n &= (1 << bit_size) - 1;
  uint64_t pos = index * bit_size;

  for (uint64_t bit = 0; bit < bit_size; ++bit) {
    uint64_t byte_idx   = (pos + bit) / 8;
    uint64_t bit_offset = (pos + bit) % 8;

    if (n & 1 << bit) {
      data[byte_idx] |= 1 << bit_offset;
    } else {
      data[byte_idx] &= ~(1 << bit_offset);
    }
  }
}

uint64_t get(const uint8_t *data, uint64_t index, uint64_t bit_size) {
  if (bit_size == 0) {
    return 0;
  }

  uint64_t pos    = index * bit_size;
  uint64_t result = 0;

  for (uint64_t bit = 0; bit < bit_size; ++bit) {
    uint64_t byte_idx   = (pos + bit) / 8;
    uint64_t bit_offset = (pos + bit) % 8;

    if (data[byte_idx] & (1 << bit_offset)) {
      result |= 1 << bit;
    }
  }

  return result;
}

}  // namespace bitmap

}  // namespace pouq
