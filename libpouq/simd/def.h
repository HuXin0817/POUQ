#pragma once

#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define POUQ_X86_ARCH
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
#define POUQ_ARM_ARCH
#include <arm_neon.h>
#else
#error "Unsupported architecture. POUQ requires x86/x64 or ARM with NEON support."
#endif

using CodeUnit = std::tuple<uint8_t, uint8_t, uint16_t>;
#ifdef POUQ_X86_ARCH
using RecPara = std::tuple<__m128, __m128>;
#elif defined(POUQ_ARM_ARCH)
using RecPara = std::tuple<float32x4_t, float32x4_t>;
#endif

inline void
set_rec_para(
    RecPara* p, float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7) {
#ifdef POUQ_X86_ARCH
  *p = {
      _mm_setr_ps(x0, x1, x2, x3),
      _mm_setr_ps(x4, x5, x6, x7),
  };
#elif defined(POUQ_ARM_ARCH)
  float lower_vals[4] = {x0, x1, x2, x3};
  float step_vals[4] = {x4, x5, x6, x7};
  *p = {
      vld1q_f32(lower_vals),
      vld1q_f32(step_vals),
  };
#endif
}
