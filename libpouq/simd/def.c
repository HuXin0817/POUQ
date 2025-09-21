#include "def.h"

void
set_rec_para(
    RecPara* p, float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7) {
  p->lower = _mm_setr_ps(x0, x1, x2, x3);
  p->step_size = _mm_setr_ps(x4, x5, x6, x7);
}
