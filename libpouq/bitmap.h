#pragma once

#include <cstdint>
#include <cstring>

namespace pouq::bitmap {

template <size_t bit_size>
void set(uint8_t *data, size_t index, size_t n) {
  static_assert(bit_size <= sizeof(size_t) * 8);
  if constexpr (bit_size == 0) {
    return;
  } else if constexpr (bit_size == 8) {
    data[index] = static_cast<uint8_t>(n);
  } else if constexpr (bit_size == 16) {
    const auto value = static_cast<uint16_t>(n);
    std::memcpy(&data[index * 2], &value, sizeof(uint16_t));
  } else if constexpr (bit_size == 32) {
    const auto value = static_cast<uint32_t>(n);
    std::memcpy(&data[index * 4], &value, sizeof(uint32_t));
  } else {
    n &= (static_cast<size_t>(1) << bit_size) - 1;
    const size_t pos = index * bit_size;
    for (size_t bit = 0; bit < bit_size; ++bit) {
      const size_t byte_idx   = (pos + bit) / 8;
      const size_t bit_offset = (pos + bit) % 8;
      if (n & 1 << bit) {
        data[byte_idx] |= 1 << bit_offset;
      } else {
        data[byte_idx] &= ~(1 << bit_offset);
      }
    }
  }
}

template <size_t bit_size>
size_t get(const uint8_t *data, size_t index) {
  static_assert(bit_size <= sizeof(size_t) * 8);
  if constexpr (bit_size == 0) {
    return 0;
  } else if constexpr (bit_size == 8) {
    return data[index];
  } else if constexpr (bit_size == 16) {
    uint16_t value;
    std::memcpy(&value, &data[index * 2], sizeof(uint16_t));
    return value;
  } else if constexpr (bit_size == 32) {
    uint32_t value;
    std::memcpy(&value, &data[index * 4], sizeof(uint32_t));
    return value;
  } else {
    const size_t pos    = index * bit_size;
    size_t       result = 0;
    for (size_t bit = 0; bit < bit_size; ++bit) {
      const size_t byte_idx   = (pos + bit) / 8;
      const size_t bit_offset = (pos + bit) % 8;
      if (data[byte_idx] & 1 << bit_offset) {
        result |= 1 << bit;
      }
    }
    return result;
  }
}

}  // namespace pouq::bitmap
