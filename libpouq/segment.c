#include "segment.h"

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
        bool do_count_freq,
        float* lowers,
        float* uppers) {
  assert(size > 0);
  assert(data_map != NULL);
  if (do_count_freq) {
    assert(freq_map != NULL);
  }

  int k = 4;
  k = min(size, k);

  int* sum_count = NULL;
  if (do_count_freq) {
    sum_count = malloc((size + 1) * sizeof(int));
    sum_count[0] = 0;
    for (int i = 1; i <= size; ++i) {
      sum_count[i] = sum_count[i - 1] + freq_map[i - 1];
    }
  }

  float* prev_dp = malloc((size + 1) * sizeof(float));
  float* curr_dp = malloc((size + 1) * sizeof(float));

  for (int i = 0; i <= size; ++i) {
    prev_dp[i] = FLT_MAX;
  }
  prev_dp[0] = 0.0f;

  int* prev_idx = malloc((size + 1) * (k + 1) * sizeof(int));
  memset(prev_idx, 0, (size + 1) * (k + 1) * sizeof(int));

  Task* tasks = malloc(size * sizeof(Task));
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
        tasks[tasks_size++] = (Task){
            .j = current_j,
            .left = mid + 1,
            .right = r,
            .opt_left = split_pos,
            .opt_right = opt_r,
        };
        tasks[tasks_size++] = (Task){
            .j = current_j,
            .left = l,
            .right = mid - 1,
            .opt_left = opt_l,
            .opt_right = split_pos,
        };
      }
    }

    float* temp_dp = prev_dp;
    prev_dp = curr_dp;
    curr_dp = temp_dp;

    for (int i = 0; i <= size; ++i) {
      curr_dp[i] = FLT_MAX;
    }
  }

  free(tasks);

  int split_pos[4];

  int curr_pos = size;
  for (int j = k; j > 0; --j) {
    int m = prev_idx[curr_pos * (k + 1) + j];
    split_pos[j - 1] = m;
    curr_pos = m;
  }

  for (int t = 0; t < k; ++t) {
    int start = split_pos[t];
    int end = (t < k - 1) ? (split_pos[t + 1] - 1) : (size - 1);

    lowers[t] = data_map[start];
    uppers[t] = data_map[end];
  }

  free(sum_count);
  free(prev_dp);
  free(curr_dp);
  free(prev_idx);

  return k;
}
