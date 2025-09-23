#pragma once

#include <immintrin.h>
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

#define do_malloc(ptr, type, size)                                         \
  do {                                                                     \
    int result = posix_memalign((void**)&ptr, 256, (size) * sizeof(type)); \
    if (result != 0) {                                                     \
      ptr = NULL;                                                          \
      goto cleanup;                                                        \
    }                                                                      \
  } while (0)

#define do_free(ptr)   \
  do {                 \
    if (ptr != NULL) { \
      free(ptr);       \
      ptr = NULL;      \
    }                  \
  } while (0)

#define min(A, B) ((A) < (B) ? (A) : (B))
#define max(A, B) ((A) > (B) ? (A) : (B))

#define rand_float(min, max) ((float)min + (float)rand() / (float)RAND_MAX * ((float)max - (float)min))

#define DIV 3.0f
