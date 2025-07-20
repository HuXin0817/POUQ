#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <cstring>
#include <immintrin.h> // AVX2 intrinsics

class Float32Quantizer
{
public:
  explicit Float32Quantizer(size_t dim) : dim(dim) {}

  void train(const float *data, size_t data_size)
  {
    encode = new float[data_size];
    memcpy(encode, data, data_size * sizeof(float));
  }

  float l2distance(const float *data, size_t data_index) const
  {
    float dis = 0.0f;
    const float *encode_ptr = encode + data_index;

    // AVX2 vectorized computation for groups of 8 floats
    size_t simd_end = (dim / 8) * 8;
    __m256 sum_vec = _mm256_setzero_ps();

    for (size_t i = 0; i < simd_end; i += 8)
    {
      // Load 8 floats from data and encode
      __m256 data_vec = _mm256_loadu_ps(&data[i]);
      __m256 encode_vec = _mm256_loadu_ps(&encode_ptr[i]);

      // Compute difference
      __m256 diff_vec = _mm256_sub_ps(data_vec, encode_vec);

      // UQuare the differences and accumulate
      sum_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_vec);
    }

    // Horizontal sum of the 8 elements in sum_vec
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128 = _mm_add_ps(sum_high, sum_low);

    // Further reduce to single value
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    dis = _mm_cvtss_f32(sum_128);

    // Handle remaining elements (if dim is not multiple of 8)
    for (size_t i = simd_end; i < dim; i++)
    {
      float diff = data[i] - encode_ptr[i];
      dis += diff * diff;
    }

    return dis;
  }

  ~Float32Quantizer() { delete[] encode; }

private:
  size_t nbit;
  size_t dim;
  float *encode = nullptr;
};
