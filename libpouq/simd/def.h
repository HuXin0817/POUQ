#pragma once

#include <stdint.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define POUQ_X86_ARCH
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
#define POUQ_ARM_ARCH
#include <arm_neon.h>
#else
#error "Unsupported architecture. POUQ requires x86/x64 or ARM with NEON support."
#endif

typedef struct {
  uint8_t x1;
  uint8_t x2;
  uint16_t code;
} CodeUnit;

#if defined(POUQ_X86_ARCH)
typedef struct {
  __m128 lower;
  __m128 step_size;
} RecPara;
#elif defined(POUQ_ARM_ARCH)
typedef struct {
  float32x4_t lower;
  float32x4_t step_size;
} RecPara;
#endif

void
set_rec_para(
    RecPara* p, float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7);

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
