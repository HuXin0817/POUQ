#pragma once

#include <omp.h>

#include "optimize.h"
#include "segment.h"
#include "util.h"

void
train_impl(int dim,
           CodeUnit* code,
           RecPara* rec_para,
           const float* data,
           int size,
           const Parameter parameter);

typedef struct {
  CodeUnit* code;
  RecPara* rec_para;
} Result;

Result
train(int dim, const float* data, int size, const Parameter parameter);

typedef struct {
  uint32_t* code;
  SQ4RecPara* rec_para;
} SQ4Result;

void
train_impl_sq4(int dim, uint32_t* code, SQ4RecPara* rec_para, const float* data, int size);

SQ4Result
train_sq4(int dim, const float* data, int size);
