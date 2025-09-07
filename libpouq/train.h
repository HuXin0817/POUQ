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

bool
train_impl(int dim,
           CodeUnit* code,
           RecPara* rec_para,
           const float* data,
           int size,
           const Parameter& parameter = Parameter());

std::pair<CodeUnit*, RecPara*>
train(int dim, const float* data, int size, const Parameter& parameter = Parameter());
