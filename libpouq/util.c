#include "util.h"

float
rand_float(float a, float b) {
  return a + (b - a) * ((float)rand() / (RAND_MAX + 1.0));
}

int
partition(float* arr, int low, int high) {
  int pivot_idx = low + rand() % (high - low + 1);
  float tmp = arr[high];
  arr[high] = arr[pivot_idx];
  arr[pivot_idx] = tmp;

  float pivot = arr[high];
  int i = low - 1;

  for (int j = low; j <= high - 1; j++) {
    if (arr[j] <= pivot) {
      i++;
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
  }
  tmp = arr[i + 1];
  arr[i + 1] = arr[high];
  arr[high] = tmp;
  return (i + 1);
}

void
sort(float* arr, int low, int high) {
  if (low < high) {
    int pi = partition(arr, low, high);
    sort(arr, low, pi - 1);
    sort(arr, pi + 1, high);
  }
}

int
get_sorted_data(const float* data, int size, int d, int dim, float* sorted_data) {
  int pos = 0;
  for (int i = d; i < size; i += dim) {
    sorted_data[pos++] = data[i];
  }
  sort(sorted_data, 0, pos);
  return pos;
}

int
count_unique(const float* arr, int len) {
  int count = 1;
  for (int i = 1; i < len; i++) {
    if (arr[i] != arr[i - 1]) {
      count++;
    }
  }
  return count;
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
