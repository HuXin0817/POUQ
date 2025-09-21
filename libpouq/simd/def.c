#include "def.h"

void
set_rec_para(
    RecPara* p, float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7) {
#if defined(POUQ_X86_ARCH)
  p->lower = _mm_setr_ps(x0, x1, x2, x3);
  p->step_size = _mm_setr_ps(x4, x5, x6, x7);
#elif defined(POUQ_ARM_ARCH)
  float lower_vals[4] = {x0, x1, x2, x3};
  float step_vals[4] = {x4, x5, x6, x7};
  p->lower = vld1q_f32(lower_vals);
  p->step_size = vld1q_f32(step_vals);
#endif
}
