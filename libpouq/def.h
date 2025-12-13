#pragma once

#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
  uint8_t x1;
  uint8_t x2;
  uint16_t code;
} CodeUnit;

typedef struct {
  __m128 lower;
  __m128 step_size;
} RecPara;

#define min(A, B) ((A) < (B) ? (A) : (B))
#define max(A, B) ((A) > (B) ? (A) : (B))

#define swap(A, B)       \
  do {                   \
    typeof(A) tmp = (A); \
    (A) = (B);           \
    (B) = tmp;           \
  } while (0)
