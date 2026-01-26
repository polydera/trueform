/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once

#include "../core/coordinate_type.hpp"
#include "../core/frame_of.hpp"
#include "../core/points_buffer.hpp"
#include "../core/policy/unwrap.hpp"
#include "../core/transformation.hpp"
#include "../spatial/nearest_neighbor.hpp"
#include "../spatial/nearest_neighbors.hpp"
#include "../spatial/neighbor_search.hpp"
#include "../spatial/policy/tree.hpp"
#include "./fit_rigid_alignment.hpp"

#include <cmath>
#include <vector>

namespace tf {

/// @ingroup geometry_registration
/// @brief Fit a rigid transformation using k-nearest neighbor correspondences.
///
/// For each point in X, finds the k nearest neighbors in Y and computes a
/// weighted correspondence point. The weights use a Gaussian kernel:
///
///   weight_j = exp(-dist_j² / (2σ²))
///
/// where σ defaults to the distance of the k-th neighbor (adaptive scaling).
///
/// This is equivalent to one iteration of ICP when k=1. For k>1, soft
/// correspondences provide robustness to noise and partial overlap.
///
/// @param X Source point set.
/// @param Y Target point set (searched for neighbors).
/// @param k Number of nearest neighbors (default: 1 = classic ICP).
/// @param sigma Gaussian kernel width. If negative, uses the k-th neighbor
///              distance as sigma (adaptive). Default: -1.
/// @return Rigid transform mapping X -> Y.
template <typename Policy0, typename Policy1>
auto fit_knn_alignment(
    const tf::points<Policy0> &X, const tf::points<Policy1> &Y,
    tf::points_buffer<tf::coordinate_type<Policy1>,
                      tf::coordinate_dims_v<Policy1>> &buffer,
    std::size_t k = 1, tf::coordinate_type<Policy0, Policy1> sigma = -1) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(Dims == tf::coordinate_dims_v<Policy1>,
                "Point sets must have the same dimensionality");
  static_assert(Dims == 2 || Dims == 3,
                "Only 2D and 3D point sets are supported");
  static_assert(tf::has_tree_policy<Policy1>,
                "Target point set Y must have a tree policy attached");
  buffer.allocate(X.size());
  if (k == 1) {
    // Classic ICP: single nearest neighbor
    tf::parallel_for_each(tf::zip(X, buffer), [&](auto tup) {
      auto &&[x, out] = tup;
      auto [id, cpt] = tf::neighbor_search(Y, tf::transformed(x, tf::frame_of(X)));
      out = cpt.point;
    });
  } else {
    // Soft correspondences: k-nearest neighbors with Gaussian weighting
    tf::parallel_for_each(tf::zip(X, buffer), [&](auto tup) {
      auto &&[x, out] = tup;
      std::array<tf::nearest_neighbor<typename Policy1::index_type,
                                      tf::coordinate_type<Policy1>,
                                      tf::coordinate_dims_v<Policy1>>,
                 10>
          knn_buffer;
      auto knn = tf::make_nearest_neighbors(knn_buffer.begin(),
                                            std::min(k, std::size_t(10)));
      tf::neighbor_search(Y, tf::transformed(x, tf::frame_of(X)),
                          knn);
      auto sig = sigma < 0 ? knn.metric() : sigma * sigma;
      for (std::size_t i = 0; i < Dims; ++i)
        out[i] = 0;
      tf::coordinate_type<Policy1> w = 0;
      for (const auto &neighbor : knn) {
        auto l_w = std::exp(-neighbor.metric() / (T(2) * sig));
        w += l_w;
        out += neighbor.info.point.as_vector_view() * l_w;
      }
      out.as_vector_view() /= w;
    });
  }

  return tf::fit_rigid_alignment(X, buffer.points());
}

template <typename Policy0, typename Policy1>
auto fit_knn_alignment(const tf::points<Policy0> &X,
                       const tf::points<Policy1> &Y, std::size_t k = 1,
                       tf::coordinate_type<Policy0, Policy1> sigma = -1) {

  tf::points_buffer<tf::coordinate_type<Policy1>,
                    tf::coordinate_dims_v<Policy1>>
      buffer;
  return fit_knn_alignment(X, Y, buffer, k, sigma);
}

} // namespace tf
