#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "optimize.h"
#include "segment.h"
#include "simd/distance.h"
#include "util.h"

void
train_impl(int dim_,
           CodeUnit* code_,
           RecPara* rec_para_,
           const float* data,
           int size,
           const Parameter& parameter = Parameter());

std::pair<CodeUnit*, RecPara*>
train(int dim_, const float* data, int size, const Parameter& parameter = Parameter());
