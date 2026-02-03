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

#include "../../core/coordinate_type.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/views/zip.hpp"
#include "../../spatial/nearest_neighbor.hpp"
#include "../../spatial/nearest_neighbors.hpp"
#include "../../spatial/neighbor_search.hpp"
#include "../../spatial/policy/tree.hpp"
#include "./fit_rigid_alignment_point_to_point.hpp"

#include <cmath>

namespace tf::geometry {

/// @brief Workspace for point-to-point kNN alignment.
template <typename T, std::size_t Dims> struct knn_alignment_point_state {
  tf::points_buffer<T, Dims> points;
};

/// @brief Fit rigid transformation using k-NN correspondences (point-to-point).
///
/// For each point in X, finds the k nearest neighbors in Y and computes a
/// weighted correspondence point. Uses point-to-point error metric.
///
/// @param X Source point set.
/// @param Y Target point set with tree (searched for neighbors).
/// @param state Reusable workspace.
/// @param k Number of nearest neighbors (default: 1 = classic ICP).
/// @param sigma Gaussian kernel width. If negative, uses adaptive scaling.
/// @return Rigid transform mapping X -> Y.
template <typename Policy0, typename Policy1>
auto fit_knn_alignment_point_to_point(
    const tf::points<Policy0> &X, const tf::points<Policy1> &Y,
    knn_alignment_point_state<tf::coordinate_type<Policy1>,
                              tf::coordinate_dims_v<Policy1>> &state,
    std::size_t k = 1, tf::coordinate_type<Policy0, Policy1> sigma = -1) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(Dims == tf::coordinate_dims_v<Policy1>,
                "Point sets must have the same dimensionality");
  static_assert(Dims == 2 || Dims == 3,
                "Only 2D and 3D point sets are supported");
  static_assert(tf::has_tree_policy<Policy1>,
                "Target point set Y must have a tree policy attached");

  state.points.allocate(X.size());

  if (k == 1) {
    // Classic ICP: single nearest neighbor
    tf::parallel_for_each(tf::zip(X, state.points), [&](auto tup) {
      auto &&[x, out] = tup;
      auto [id, cpt] =
          tf::neighbor_search(Y, tf::transformed(x, tf::frame_of(X)));
      out = cpt.point;
    });
  } else {
    // Soft correspondences: k-nearest neighbors with Gaussian weighting
    tf::parallel_for_each(tf::zip(X, state.points), [&](auto tup) {
      constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
      auto &&[x, out] = tup;
      std::array<tf::nearest_neighbor<typename Policy1::index_type,
                                      tf::coordinate_type<Policy1>, Dims>,
                 10>
          knn_buffer;
      auto knn = tf::make_nearest_neighbors(knn_buffer.begin(),
                                            std::min(k, std::size_t(10)));
      tf::neighbor_search(Y, tf::transformed(x, tf::frame_of(X)), knn);

      auto sig = sigma < 0 ? knn.metric() : sigma * sigma;
      out = tf::zero;
      T w = 0;

      for (const auto &neighbor : knn) {
        auto l_w = std::exp(-neighbor.metric() / (T(2) * sig));
        w += l_w;
        out += neighbor.info.point.as_vector_view() * l_w;
      }
      out.as_vector_view() /= w;
    });
  }

  return tf::geometry::fit_rigid_alignment_point_to_point(X,
                                                          state.points.points());
}

/// @brief Fit rigid transformation using k-NN (point-to-point, allocates
/// internally).
template <typename Policy0, typename Policy1>
auto fit_knn_alignment_point_to_point(
    const tf::points<Policy0> &X, const tf::points<Policy1> &Y,
    std::size_t k = 1, tf::coordinate_type<Policy0, Policy1> sigma = -1) {
  knn_alignment_point_state<tf::coordinate_type<Policy1>,
                            tf::coordinate_dims_v<Policy1>>
      state;
  return fit_knn_alignment_point_to_point(X, Y, state, k, sigma);
}

} // namespace tf::geometry
