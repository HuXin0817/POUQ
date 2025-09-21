#pragma once

#include <omp.h>

#include "optimize.h"
#include "segment.h"
#include "simd/def.h"
#include "simd/distance.h"
#include "util.h"

int
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
