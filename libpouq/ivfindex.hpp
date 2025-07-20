#pragma once

#include "quantizer.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

struct ClusterNode {
  std::vector<float>  centroid;
  std::vector<size_t> indices;
};

inline std::vector<ClusterNode> centroids_;

template <typename Quantizer>
class IVFIndex {

public:
  IVFIndex(size_t nlist, size_t dim) : candidate_centroids(nlist), nlist_(nlist), dim_(dim), quantizer_(dim_) {
    if (centroids_.empty()) {}

    sum_result.reserve(1024 * 1024);
  }

  void train(const float *data, size_t size) {
    quantizer_.train(data, size);
    size_t num_samples = size / dim_;
    if (centroids_.empty()) {
      centroids_.resize(nlist_);
      for (auto &centroid : centroids_) {
        centroid.centroid.resize(dim_);
      }
      kmeans_clustering(data, num_samples);
    }
  }

  static bool leq(const std::pair<size_t, float> &p1, const std::pair<size_t, float> &p2) {
    return p1.second < p2.second;
  }

  static bool req(const std::pair<size_t, float> &p1, const std::pair<size_t, float> &p2) {
    return p1.second > p2.second;
  }

  std::vector<std::pair<size_t, float>> search(const float *query, size_t k, size_t nprobe) {
    nprobe = std::min(nprobe, nlist_);
#pragma omp parallel for
    for (size_t i = 0; i < nlist_; ++i) {
      candidate_centroids[i] = {i, l2distance(centroids_[i].centroid, query, dim_)};
    }

    std::nth_element(
        candidate_centroids.begin(), candidate_centroids.begin() + nprobe - 1, candidate_centroids.end(), leq);
    std::vector<std::pair<size_t, float>>().swap(sum_result);
#pragma omp parallel for
    for (size_t i = 0; i < nprobe; i++) {
      const auto                            centroid_idx_ = candidate_centroids[i].first;
      std::vector<std::pair<size_t, float>> results;

      const auto &vector_indices = centroids_[centroid_idx_].indices;
      results.reserve(vector_indices.size());

      for (const auto vec_idx : vector_indices) {
        results.emplace_back(vec_idx, quantizer_.l2distance(query, dim_ * vec_idx));
      }

#pragma omp critical
      { sum_result.insert(sum_result.end(), results.begin(), results.end()); }
    }

    k = std::min(k, sum_result.size());
    if (k > 0) {
      std::nth_element(sum_result.begin(), sum_result.begin() + k - 1, sum_result.end(), leq);
    }

    return sum_result;
  }

private:
  size_t    nlist_;
  size_t    dim_;
  Quantizer quantizer_;

  std::vector<std::pair<size_t, float>> sum_result;
  std::vector<std::pair<size_t, float>> candidate_centroids;

  void kmeans_clustering(const float *data, size_t num_samples) {
    constexpr size_t max_iterations = 500;
    constexpr float  tolerance      = 1e-9f;

    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_int_distribution<size_t> dis(0, num_samples - 1);
    std::uniform_real_distribution<float> real_dis(0.0f, 1.0f);

    size_t       first_idx = dis(gen);
    const float *first_vec = data + first_idx * dim_;
    memcpy(centroids_[0].centroid.data(), first_vec, dim_ * sizeof(float));
    centroids_[0].indices.clear();

    for (size_t i = 1; i < nlist_; ++i) {
      std::vector<float> distances(num_samples);
      float              total_distance = 0.0f;

#pragma omp parallel for reduction(+ : total_distance)
      for (size_t j = 0; j < num_samples; ++j) {
        const float *vec      = data + j * dim_;
        float        min_dist = std::numeric_limits<float>::max();

        for (size_t k = 0; k < i; ++k) {
          float dist = l2distance(vec, centroids_[k].centroid, dim_);
          min_dist   = std::min(min_dist, dist);
        }

        distances[j] = min_dist * min_dist;
        total_distance += distances[j];
      }

      float  random_val      = real_dis(gen) * total_distance;
      float  cumulative_prob = 0.0f;
      size_t selected_idx    = 0;

      for (size_t j = 0; j < num_samples; ++j) {
        cumulative_prob += distances[j];
        if (cumulative_prob >= random_val) {
          selected_idx = j;
          break;
        }
      }

      const float *selected_vec = data + selected_idx * dim_;
      memcpy(centroids_[i].centroid.data(), selected_vec, dim_ * sizeof(float));
      centroids_[i].indices.clear();
    }

    std::vector<omp_lock_t> locks(nlist_);
    for (auto &lock : locks) {
      omp_init_lock(&lock);
    }

    size_t consecutive_converged = 0;
    for (size_t iter = 0; iter < max_iterations; ++iter) {
      if (iter % 50 == 0) {
        std::cout << "[" << iter << "/" << max_iterations << "] " << std::endl;
      }

      for (auto &centroid : centroids_) {
        centroid.indices.clear();
      }

#pragma omp parallel for
      for (size_t i = 0; i < num_samples; ++i) {
        const float *vec          = data + i * dim_;
        size_t       best_cluster = 0;
        float        min_distance = std::numeric_limits<float>::max();

        for (size_t j = 0; j < nlist_; ++j) {
          float distance = l2distance(vec, centroids_[j].centroid, dim_);
          if (distance < min_distance) {
            min_distance = distance;
            best_cluster = j;
          }
        }

        omp_set_lock(&locks[best_cluster]);
        centroids_[best_cluster].indices.push_back(i);
        omp_unset_lock(&locks[best_cluster]);
      }

      bool not_converged = false;
#pragma omp parallel for reduction(|| : not_converged)
      for (size_t c = 0; c < nlist_; ++c) {
        auto &centroid = centroids_[c];
        if (centroid.indices.empty()) {

          size_t random_idx;
#pragma omp critical
          { random_idx = dis(gen); }
          const float *random_vec = data + random_idx * dim_;
          std::copy_n(random_vec, dim_, centroid.centroid.data());
          not_converged = true;
          continue;
        }

        std::vector<float> new_centroid(dim_, 0.0f);
        for (const auto idx : centroid.indices) {
          const float *vec = data + idx * dim_;
          for (size_t j = 0; j < dim_; ++j) {
            new_centroid[j] += vec[j];
          }
        }

        for (size_t j = 0; j < dim_; ++j) {
          new_centroid[j] /= static_cast<float>(centroid.indices.size());
        }

        float centroid_shift = l2distance(centroid.centroid, new_centroid, dim_);
        if (centroid_shift > tolerance) {
          not_converged = true;
        }
        centroid.centroid = std::move(new_centroid);
      }

      if (not_converged) {
        consecutive_converged = 0;
      } else {
        consecutive_converged++;
      }

      if (consecutive_converged >= 16) {
        break;
      }
    }

    for (auto &lock : locks) {
      omp_destroy_lock(&lock);
    }
  }
};

using IVF     = IVFIndex<Float32Quantizer>;
using IVFUQ4  = IVFIndex<UQ4bitSIMDQuantizer>;
using IVFPOUQ = IVFIndex<POUQ4bitSIMDQuantizer>;
