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
#include "../../core/policy/normals.hpp"
#include "../../core/unit_vectors_buffer.hpp"
#include "../../core/views/zip.hpp"
#include "../../spatial/nearest_neighbor.hpp"
#include "../../spatial/nearest_neighbors.hpp"
#include "../../spatial/neighbor_search.hpp"
#include "../../spatial/policy/tree.hpp"
#include "./fit_rigid_alignment_point_to_plane.hpp"

#include <cmath>

namespace tf::geometry {

/// @brief Workspace for point-to-plane kNN alignment.
template <typename T, std::size_t Dims> struct knn_alignment_plane_state {
  tf::points_buffer<T, Dims> points;
  tf::unit_vectors_buffer<T, Dims> normals;
  plane_alignment_state<T> alignment_state;
};

/// @brief Fit rigid transformation using k-NN correspondences (point-to-plane).
///
/// For each point in X, finds the k nearest neighbors in Y and computes a
/// weighted correspondence point and normal. Uses point-to-plane error metric
/// for faster ICP convergence.
///
/// @param X Source point set.
/// @param Y Target point set with tree and normals.
/// @param state Reusable workspace.
/// @param k Number of nearest neighbors (default: 1 = classic ICP).
/// @param sigma Gaussian kernel width. If negative, uses adaptive scaling.
/// @return Rigid transform mapping X -> Y.
template <typename Policy0, typename Policy1>
auto fit_knn_alignment_point_to_plane(
    const tf::points<Policy0> &X, const tf::points<Policy1> &Y,
    knn_alignment_plane_state<tf::coordinate_type<Policy1>,
                              tf::coordinate_dims_v<Policy1>> &state,
    std::size_t k = 1, tf::coordinate_type<Policy0, Policy1> sigma = -1) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(Dims == 3, "Point-to-plane alignment requires 3D points");
  static_assert(Dims == tf::coordinate_dims_v<Policy1>,
                "Point sets must have the same dimensionality");
  static_assert(tf::has_tree_policy<Policy1>,
                "Target point set Y must have a tree policy attached");
  static_assert(tf::has_normals_policy<Policy1>,
                "Target point set Y must have normals attached for "
                "point-to-plane");

  const auto &Y_normals = Y.normals();

  state.points.allocate(X.size());
  state.normals.allocate(X.size());

  if (k == 1) {
    // Classic ICP: single nearest neighbor
    tf::parallel_for_each(
        tf::zip(X, state.points, state.normals), [&](auto tup) {
          auto &&[x, out_pt, out_n] = tup;
          auto [id, cpt] =
              tf::neighbor_search(Y, tf::transformed(x, tf::frame_of(X)));
          out_pt = cpt.point;
          out_n = Y_normals[id];
        });
  } else {
    // Soft correspondences: k-nearest neighbors with Gaussian weighting
    tf::parallel_for_each(
        tf::zip(X, state.points, state.normals), [&](auto tup) {
          constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
          auto &&[x, out_pt, out_n] = tup;
          std::array<tf::nearest_neighbor<typename Policy1::index_type,
                                          tf::coordinate_type<Policy1>, Dims>,
                     10>
              knn_buffer;
          auto knn = tf::make_nearest_neighbors(knn_buffer.begin(),
                                                std::min(k, std::size_t(10)));
          tf::neighbor_search(Y, tf::transformed(x, tf::frame_of(X)), knn);

          auto sig = sigma < 0 ? knn.metric() : sigma * sigma;
          out_pt = tf::zero;
          tf::vector<T, Dims> normal_sum = tf::zero;
          T w = 0;

          for (const auto &neighbor : knn) {
            auto l_w = std::exp(-neighbor.metric() / (T(2) * sig));
            w += l_w;
            out_pt += neighbor.info.point.as_vector_view() * l_w;
            normal_sum += Y_normals[neighbor.element] * l_w;
          }

          out_pt.as_vector_view() /= w;
          out_n = tf::make_unit_vector(normal_sum);
        });
  }

  auto Y_with_normals =
      state.points.points() | tf::tag_normals(state.normals.unit_vectors());
  return tf::geometry::fit_rigid_alignment_point_to_plane(
      X, Y_with_normals, state.alignment_state);
}

/// @brief Fit rigid transformation using k-NN (point-to-plane, allocates
/// internally).
template <typename Policy0, typename Policy1>
auto fit_knn_alignment_point_to_plane(
    const tf::points<Policy0> &X, const tf::points<Policy1> &Y,
    std::size_t k = 1, tf::coordinate_type<Policy0, Policy1> sigma = -1) {
  knn_alignment_plane_state<tf::coordinate_type<Policy1>,
                            tf::coordinate_dims_v<Policy1>>
      state;
  return fit_knn_alignment_point_to_plane(X, Y, state, k, sigma);
}

} // namespace tf::geometry
