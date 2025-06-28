#pragma once

#include "bitmap.hpp"
#include "quantizer.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

class IvfIndex {
  struct ClusterNode {
    std::vector<float>  centroid;
    std::vector<size_t> indices;
  };

public:
  IvfIndex(size_t nlist, size_t dim) : nlist_(nlist), dim_(dim), quantizer_(4, 4, dim_) {
    centroids_.resize(nlist_);
    for (auto &centroid : centroids_) {
      centroid.centroid.resize(dim_);
    }
  }

  void train(float *data, size_t size) {
    quantizer_.train(data, size);

    size_t num_samples = size / dim_;

    size_t p = 0;
    for (size_t i = 0; i < num_samples; i++) {
      centroids_[p % nlist_].indices.push_back(i);
      p++;
    }

    for (auto &centroid : centroids_) {
      for (const auto idx : centroid.indices) {
        const auto vec = data + idx * dim_;
        for (size_t j = 0; j < dim_; j++) {
          centroid.centroid[j] += vec[j];
        }
      }

      for (auto &d : centroid.centroid) {
        d /= static_cast<float>(centroid.indices.size());
      }
    }
  }

  std::vector<size_t> search(float *query, size_t k, size_t nprobe) const {
    auto cmp = [](const std::pair<size_t, float> &p1, const std::pair<size_t, float> &p2) {
      return p1.second < p2.second;
    };

    std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t, float>>, decltype(cmp)>
        candidate_centroids(cmp);
    for (size_t i = 0; i < nlist_; ++i) {
      candidate_centroids.emplace(i, l2distance(centroids_[i].centroid, query, dim_));
    }

    nprobe = std::min(nprobe, nlist_);
    std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t, float>>, decltype(cmp)> results(cmp);
    for (size_t i = 0; i < nprobe; ++i) {
      auto centroid_idx_ = candidate_centroids.top().first;
      candidate_centroids.pop();

      const auto &vector_indices = centroids_[centroid_idx_].indices;
      for (const auto vec_idx : vector_indices) {
        float dis = 0.0f;
        for (size_t j = 0; j < dim_; ++j) {
          const float diff = query[j] - quantizer_[vec_idx * dim_ + j];
          dis += diff * diff;
        }
        results.emplace(vec_idx, dis);
      }
    }

    std::vector<size_t> indices;
    indices.reserve(k);
    while (indices.size() < k && !results.empty()) {
      indices.emplace_back(results.top().first);
      results.pop();
    }

    return indices;
  }

private:
  size_t                   nlist_;
  size_t                   dim_;
  std::vector<ClusterNode> centroids_;
  plsq::PLSQQuantizer      quantizer_;
};