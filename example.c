#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libpouq/distance.h"
#include "libpouq/train.h"

// 计算大于等于dim且是8的倍数的最小维度
int
get_padded_dim(int dim) {
  if (dim % 8 == 0) {
    return dim;
  } else {
    return (dim / 8 + 1) * 8;
  }
}

// 读取fvecs文件并将维度补成8的倍数（补0）
// 参数：
//   filename: 文件名
//   orig_dim: 输出，原始向量维度
//   new_dim: 输出，补0后的维度（8的倍数）
//   num_vec: 输出，向量数量
// 返回：
//   成功：补0后的连续float数组（大小：num_vec * new_dim）
//   失败：NULL
float*
read_fvecs_padded(const char* filename, int* orig_dim, int* new_dim, int* num_vec) {
  // 1. 先读取原始数据
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    perror("无法打开文件");
    return NULL;
  }

  // 读取原始维度
  int32_t first_dim;
  if (fread(&first_dim, sizeof(int32_t), 1, fp) != 1) {
    perror("读取维度失败");
    fclose(fp);
    return NULL;
  }
  *orig_dim = first_dim;

  // 计算单个原始向量的字节大小
  size_t vec_byte_size = sizeof(int32_t) + *orig_dim * sizeof(float);

  // 获取向量数量
  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  *num_vec = file_size / vec_byte_size;
  fseek(fp, 0, SEEK_SET);

  // 2. 计算补0后的目标维度
  *new_dim = get_padded_dim(*orig_dim);

  // 3. 分配补0后的总内存（num_vec * new_dim）
  float* padded_vecs = (float*)malloc(*num_vec * *new_dim * sizeof(float));
  if (!padded_vecs) {
    perror("内存分配失败");
    fclose(fp);
    return NULL;
  }

  // 4. 逐个读取向量并补0
  for (int i = 0; i < *num_vec; i++) {
    // 读取当前向量的维度（验证一致性）
    int32_t current_dim;
    if (fread(&current_dim, sizeof(int32_t), 1, fp) != 1) {
      perror("读取维度失败");
      free(padded_vecs);
      fclose(fp);
      return NULL;
    }
    if (current_dim != *orig_dim) {
      fprintf(stderr, "向量维度不一致（第%d个向量）\n", i);
      free(padded_vecs);
      fclose(fp);
      return NULL;
    }

    // 计算当前向量在补0数组中的起始位置
    float* dest = padded_vecs + i * *new_dim;

    // 读取原始数据到补0数组的前orig_dim个位置
    if (fread(dest, sizeof(float), *orig_dim, fp) != *orig_dim) {
      perror("读取向量数据失败");
      free(padded_vecs);
      fclose(fp);
      return NULL;
    }

    // 剩余位置补0（从orig_dim到new_dim-1）
    if (*new_dim > *orig_dim) {
      memset(dest + *orig_dim, 0, (*new_dim - *orig_dim) * sizeof(float));
    }
  }

  fclose(fp);
  return padded_vecs;
}

int
main(int argc, char* argv[]) {
  const char* filename = argv[1];

  int ReadDim, Dim = 0, N = 0;
  float* data = read_fvecs_padded(filename, &ReadDim, &Dim, &N);
  N *= Dim;

  Parameter param = {
      .max_iter = 100,
      .particle_count = 50,
      .scale_factor = 0.1f,
      .init_inertia = 0.9f,
      .final_inertia = 0.4f,
      .c1 = 1.5f,
      .c2 = 1.5f,
  };

  Result result = train(Dim, data, N, param);
  if (!result.code || !result.rec_para) {
    free(data);
    printf("train error\n");
    return 1;
  }

  float mse = 0.0f;
#pragma omp parallel for reduction(+ : mse)
  for (int i = 0; i < N; i += Dim) {
    mse += distance(Dim, result.code, result.rec_para, data + i, i);
  }
  printf("%f\n", mse / N);

  free(result.code);
  free(result.rec_para);
  free(data);
  return 0;
}
