#pragma once

#include <assert.h>

#include "def.h"

float
distance(int dim, const CodeUnit* code, const RecPara* rec_para, const float* data, int offset);

void
decode(int dim, const CodeUnit* code, const RecPara* rec_para, float* dist, int offset);

float
distance_sq4(
    int dim, const uint32_t* code, const SQ4RecPara* rec_para, const float* data, int offset);

void
decode_sq4(int dim, const uint32_t* code, const SQ4RecPara* rec_para, float* dist, int offset);
