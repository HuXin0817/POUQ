#pragma once

#include <assert.h>

#include "def.h"

float
distance(int dim, const CodeUnit* code, const RecPara* rec_para, const float* data, int offset);

void
decode(int dim, const CodeUnit* code, const RecPara* rec_para, float* dist, int offset);
