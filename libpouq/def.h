#pragma once

#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  uint8_t x1;
  uint8_t x2;
  uint16_t code;
} CodeUnit;

typedef struct {
  __m128 lower;
  __m128 step_size;
} RecPara;

#define do_malloc(ptr, size)                                                \
  do {                                                                      \
    if (posix_memalign((void**)&ptr, 256, (size) * sizeof(typeof(*ptr)))) { \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

#define min(A, B) ((A) < (B) ? (A) : (B))
#define max(A, B) ((A) > (B) ? (A) : (B))

#define rand_float(l, u) ((float)(l) + (float)rand() / (float)RAND_MAX * ((float)(u) - (float)(l)))

#define DIV 3.0f
