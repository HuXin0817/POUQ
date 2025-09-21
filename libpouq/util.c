#include "util.h"

int
compare_floats(const void* a, const void* b) {
  float fa = *(const float*)a;
  float fb = *(const float*)b;
  if (fa < fb) {
    return -1;
  }
  if (fa > fb) {
    return 1;
  }
  return 0;
}

int
get_sorted_data(const float* data, int size, int d, int dim, float* sorted_data) {
  int pos = 0;
  for (int i = d; i < size; i += dim) {
    sorted_data[pos++] = data[i];
  }

  qsort(sorted_data, pos, sizeof(float), compare_floats);

  return pos;
}

int
count_freq(const float* sorted_data, int sorted_data_size, float* data_map, int* freq_map) {
  int pos = 0;
  float curr_value = sorted_data[0];
  int count = 1;
  for (int i = 1; i < sorted_data_size; i++) {
    if (sorted_data[i] == curr_value) {
      count++;
    } else {
      data_map[pos] = curr_value;
      freq_map[pos] = count;
      pos++;

      curr_value = sorted_data[i];
      count = 1;
    }
  }

  data_map[pos] = curr_value;
  freq_map[pos] = count;
  pos++;
  return pos;
}
