#include "segment.h"

#define K 4

typedef struct {
  int j;
  int left;
  int right;
  int opt_left;
  int opt_right;
} Task;

int
segment(const float* data_map,
        const int* freq_map,
        int size,
        int do_count_freq,
        float* lowers,
        float* uppers) {
  assert(size > 0);
  assert(data_map != NULL);
  if (do_count_freq) {
    assert(freq_map != NULL);
  }

  int k = K;
  k = min(size, k);

  int* sum_count = NULL;
  if (do_count_freq) {
    do_malloc(sum_count, int, size + 1);
    sum_count[0] = 0;
    for (int i = 1; i <= size; ++i) {
      sum_count[i] = sum_count[i - 1] + freq_map[i - 1];
    }
  }

  float* prev_dp = NULL;
  float* curr_dp = NULL;
  do_malloc(prev_dp, float, size + 1);
  do_malloc(curr_dp, float, size + 1);

  for (int i = 0; i <= size; ++i) {
    prev_dp[i] = FLT_MAX;
  }
  prev_dp[0] = 0.0f;

  int* prev_idx = NULL;
  do_malloc(prev_idx, int, (size + 1) * (k + 1));
  memset(prev_idx, 0, (size + 1) * (k + 1) * sizeof(int));

  Task* tasks = NULL;
  do_malloc(tasks, Task, size);
  int tasks_size = 0;

  for (int j = 1; j <= k; ++j) {
    tasks[0].j = j;
    tasks[0].left = j;
    tasks[0].right = size;
    tasks[0].opt_left = 0;
    tasks[0].opt_right = size - 1;
    tasks_size = 1;

    while (tasks_size) {
      Task current_task = tasks[tasks_size - 1];
      tasks_size--;

      int current_j = current_task.j;
      int l = current_task.left;
      int r = current_task.right;
      int opt_l = current_task.opt_left;
      int opt_r = current_task.opt_right;

      if (l > r) {
        continue;
      }

      int mid = (l + r) / 2;
      int start = max(current_j - 1, opt_l);
      int end = min(mid - 1, opt_r);
      float min_cost = FLT_MAX;
      int split_pos = 0;

      for (int m = start; m <= end; ++m) {
        float width = data_map[mid - 1] - data_map[m];
        float cost = prev_dp[m];
        if (do_count_freq) {
          int count = sum_count[mid] - sum_count[m];
          cost += width * width * (float)(count);
        } else {
          cost += width * width;
        }
        if (cost < min_cost) {
          min_cost = cost;
          split_pos = m;
        }
      }

      curr_dp[mid] = min_cost;

      prev_idx[mid * (k + 1) + current_j] = split_pos;

      if (l < r) {
        tasks[tasks_size].j = current_j;
        tasks[tasks_size].left = mid + 1;
        tasks[tasks_size].right = r;
        tasks[tasks_size].opt_left = split_pos;
        tasks[tasks_size].opt_right = opt_r;
        tasks_size++;
        tasks[tasks_size].j = current_j;
        tasks[tasks_size].left = l;
        tasks[tasks_size].right = mid - 1;
        tasks[tasks_size].opt_left = opt_l;
        tasks[tasks_size].opt_right = split_pos;
        tasks_size++;
      }
    }

    float* temp_dp = prev_dp;
    prev_dp = curr_dp;
    curr_dp = temp_dp;

    for (int i = 0; i <= size; ++i) {
      curr_dp[i] = FLT_MAX;
    }
  }

  do_free(tasks);

  int split_pos[K];

  int curr_pos = size;
  for (int j = k; j > 0; --j) {
    int m = prev_idx[curr_pos * (k + 1) + j];
    split_pos[j - 1] = m;
    curr_pos = m;
  }

  for (int t = 0; t < k; ++t) {
    int start = split_pos[t];
    int end = (t < k - 1) ? (split_pos[t + 1] - 1) : (size - 1);

    if (start < 0 || start >= size || end < 0 || end >= size) {
      goto cleanup;
    }

    lowers[t] = data_map[start];
    uppers[t] = data_map[end];
  }

cleanup:
  do_free(sum_count);
  do_free(prev_dp);
  do_free(curr_dp);
  do_free(prev_idx);

  return k;
}
