#pragma once

#include <cmath>
#include <limits>
#include <vector>

class SQQuantizer {
public:
  explicit SQQuantizer(size_t dim) : dim(dim) {}

  void train(const float *data, size_t data_size) {
    encode    = new uint8_t[(data_size + 1) / 2];  // 每个byte存储2个4bit值
    codebook  = new std::pair<float, float>[dim];
    float div = (1 << 4) - 1;

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
        codebook[i].second = (codebook[i].second - codebook[i].first) / div;
      }
      for (size_t j = i; j < data_size; j += dim) {
        set(encode, j, std::round((data[j] - codebook[i].first) / codebook[i].second));
      }
    }
  }

  float l2distance(const float *data, size_t data_index) const {
    float dis = 0.0f;
    size_t i = 0;
    
    // 两个两个地处理，优化get操作
    for (; i + 1 < dim; i += 2) {
      const size_t byte_idx = (data_index + i) / 2;
      const uint8_t byte_val = encode[byte_idx];
      
      // 提取两个4位值
      uint8_t val1, val2;
      if ((data_index + i) & 1) {
        // 起始索引为奇数
        val1 = (byte_val >> 4) & 0x0F;  // 高4位
        val2 = encode[byte_idx + 1] & 0x0F;  // 下一个字节的低4位
      } else {
        // 起始索引为偶数
        val1 = byte_val & 0x0F;          // 低4位
        val2 = (byte_val >> 4) & 0x0F;   // 高4位
      }
      
      // 计算第一个维度的距离
      float diff1 = static_cast<float>(val1) * codebook[i].second + codebook[i].first - data[i];
      dis += diff1 * diff1;
      
      // 计算第二个维度的距离
      float diff2 = static_cast<float>(val2) * codebook[i + 1].second + codebook[i + 1].first - data[i + 1];
      dis += diff2 * diff2;
    }
    
    // 处理剩余的维度（如果dim是奇数）
    if (i < dim) {
      float diff = static_cast<float>(get(encode, data_index + i)) * codebook[i].second + codebook[i].first - data[i];
      dis += diff * diff;
    }
    
    return dis;
  }

private:
  size_t                   dim;
  std::pair<float, float> *codebook = nullptr;
  uint8_t                 *encode   = nullptr;

  // 优化的set方法：直接操作半字节
  void set(uint8_t *data, size_t index, uint8_t value) {
    value &= 0x0F;  // 确保只有4bit
    const size_t byte_idx = index / 2;
    if (index & 1) {
      // 奇数索引：存储在高4位
      data[byte_idx] = (data[byte_idx] & 0x0F) | (value << 4);
    } else {
      // 偶数索引：存储在低4位
      data[byte_idx] = (data[byte_idx] & 0xF0) | value;
    }
  }

  // 优化的get方法：直接读取半字节
  uint8_t get(const uint8_t *data, size_t index) const {
    const size_t byte_idx = index / 2;
    if (index & 1) {
      // 奇数索引：从高4位读取
      return (data[byte_idx] >> 4) & 0x0F;
    } else {
      // 偶数索引：从低4位读取
      return data[byte_idx] & 0x0F;
    }
  }
};
