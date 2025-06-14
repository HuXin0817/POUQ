#pragma once

#include <cmath>
#include <limits>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

class SQ8Quantizer {
public:
  explicit SQ8Quantizer(size_t dim) : dim(dim) {}

  void train(const float *data, size_t data_size) {
    encode   = new uint8_t[data_size];
    codebook = new std::pair<float, float>[dim];

#pragma omp parallel for
    for (size_t i = 0; i < dim; i++) {
      codebook[i].first  = std::numeric_limits<float>::min();
      codebook[i].second = std::numeric_limits<float>::max();
      for (size_t j = i; j < data_size; j += dim) {
        codebook[i].first  = std::max(codebook[i].first, data[j]);
        codebook[i].second = std::min(codebook[i].second, data[j]);
      }
      if (codebook[i].first == codebook[i].second) {
        codebook[i].second = 1.0f;
      } else {
        codebook[i].second = (codebook[i].second - codebook[i].first) / 255.0f;
      }
      for (size_t j = i; j < data_size; j += dim) {
        encode[j] = std::round((data[j] - codebook[i].first) / codebook[i].second);
      }
    }
  }

  float l2distance(const float *data, size_t data_index) const {
#ifdef __AVX2__
    return l2distance_avx2(data, data_index);
#elif defined(__ARM_NEON)
    return l2distance_neon(data, data_index);
#else
    return l2distance_scalar(data, data_index);
#endif
  }

private:
  size_t                   dim;
  std::pair<float, float> *codebook = nullptr;
  uint8_t                 *encode   = nullptr;

  float l2distance_scalar(const float *data, size_t data_index) const {
    float dis = 0.0f;
    for (size_t i = 0; i < dim; i++) {
      float diff = static_cast<float>(encode[data_index + i]) * codebook[i].second + codebook[i].first - data[i];
      dis += diff * diff;
    }
    return dis;
  }

#ifdef __AVX2__
  float l2distance_avx2(const float *data, size_t data_index) const {
    __m256 sum      = _mm256_setzero_ps();
    size_t simd_end = (dim / 8) * 8;

    for (size_t i = 0; i < simd_end; i += 8) {
      __m128i encoded_u8  = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&encode[data_index + i]));
      __m256i encoded_u32 = _mm256_cvtepu8_epi32(encoded_u8);
      __m256  encoded_f   = _mm256_cvtepi32_ps(encoded_u32);

      __m256 scales  = _mm256_set_ps(codebook[i + 7].second,
          codebook[i + 6].second,
          codebook[i + 5].second,
          codebook[i + 4].second,
          codebook[i + 3].second,
          codebook[i + 2].second,
          codebook[i + 1].second,
          codebook[i + 0].second);
      __m256 offsets = _mm256_set_ps(codebook[i + 7].first,
          codebook[i + 6].first,
          codebook[i + 5].first,
          codebook[i + 4].first,
          codebook[i + 3].first,
          codebook[i + 2].first,
          codebook[i + 1].first,
          codebook[i + 0].first);

      __m256 query = _mm256_loadu_ps(&data[i]);

      __m256 reconstructed = _mm256_fmadd_ps(encoded_f, scales, offsets);
      __m256 diff          = _mm256_sub_ps(reconstructed, query);

      sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low  = _mm256_castps256_ps128(sum);
    __m128 sum_128  = _mm_add_ps(sum_low, sum_high);
    __m128 sum_64   = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
    __m128 sum_32   = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));
    float  result   = _mm_cvtss_f32(sum_32);

    for (size_t i = simd_end; i < dim; i++) {
      float diff = static_cast<float>(encode[data_index + i]) * codebook[i].second + codebook[i].first - data[i];
      result += diff * diff;
    }

    return result;
  }
#endif

#ifdef __ARM_NEON
  float l2distance_neon(const float *data, size_t data_index) const {
    float32x4_t sum      = vdupq_n_f32(0.0f);
    size_t      simd_end = (dim / 4) * 4;

    for (size_t i = 0; i < simd_end; i += 4) {
      uint8x8_t   encoded_u8  = vld1_u8(&encode[data_index + i]);
      uint16x4_t  encoded_u16 = vget_low_u16(vmovl_u8(encoded_u8));
      uint32x4_t  encoded_u32 = vmovl_u16(encoded_u16);
      float32x4_t encoded_f   = vcvtq_f32_u32(encoded_u32);

      float32x4_t scales = {codebook[i].second, codebook[i + 1].second, codebook[i + 2].second, codebook[i + 3].second};
      float32x4_t offsets = {codebook[i].first, codebook[i + 1].first, codebook[i + 2].first, codebook[i + 3].first};

      float32x4_t query = vld1q_f32(&data[i]);

      float32x4_t reconstructed = vmlaq_f32(offsets, encoded_f, scales);
      float32x4_t diff          = vsubq_f32(reconstructed, query);

      sum = vmlaq_f32(sum, diff, diff);
    }

    float32x2_t sum_pair = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float       result   = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    for (size_t i = simd_end; i < dim; i++) {
      float diff = static_cast<float>(encode[data_index + i]) * codebook[i].second + codebook[i].first - data[i];
      result += diff * diff;
    }

    return result;
  }
#endif
};
