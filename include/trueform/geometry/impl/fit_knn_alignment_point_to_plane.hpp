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

#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/policy/normals.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/take.hpp"
#include "../../core/views/zip.hpp"
#include "../../spatial/nearest_neighbor.hpp"
#include "../../spatial/nearest_neighbors.hpp"
#include "../../spatial/neighbor_search.hpp"
#include "../../spatial/policy/tree.hpp"
#include "../knn_alignment_config.hpp"
#include "../knn_alignment_state.hpp"
#include "./fit_rigid_alignment_point_to_plane.hpp"
#include "tbb/parallel_sort.h"

#include <cmath>

namespace tf::geometry {

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

  state.target_points.allocate(X.size());
  state.target_normals.allocate(X.size());

  if (k == 1) {
    tf::parallel_for_each(
        tf::zip(X, state.target_points, state.target_normals),
        [&](auto tup) {
          auto &&[x, out_pt, out_n] = tup;
          auto [id, cpt] =
              tf::neighbor_search(Y, tf::transformed(x, tf::frame_of(X)));
          out_pt = cpt.point;
          out_n = tf::transformed_normal(Y_normals[id], tf::frame_of(Y));
        },
        tf::checked);
  } else {
    tf::parallel_for_each(
        tf::zip(X, state.target_points, state.target_normals),
        [&](auto tup) {
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
          out_n = tf::transformed_normal(tf::make_unit_vector(normal_sum),
                                         tf::frame_of(Y));
        },
        tf::checked);
  }

  auto Y_with_normals = state.target_points.points() |
                        tf::tag_normals(state.target_normals.unit_vectors());
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

/// @brief Fit rigid transformation with outlier rejection (point-to-plane).
///
/// Computes correspondences, sorts by distance, and fits using only the best
/// (1 - outlier_proportion) correspondences. Uses indices to preserve source
/// policies through filtering.
///
/// @param X Source point set.
/// @param Y Target point set with tree and normals.
/// @param state Reusable workspace.
/// @param config Configuration with outlier_proportion.
/// @return Rigid transform mapping X -> Y.
template <typename Policy0, typename Policy1>
auto fit_knn_alignment_point_to_plane(
    const tf::points<Policy0> &X, const tf::points<Policy1> &Y,
    knn_alignment_plane_state<tf::coordinate_type<Policy1>,
                              tf::coordinate_dims_v<Policy1>> &state,
    const tf::knn_alignment_config &config) {
  // No outlier rejection - use simple path
  if (config.outlier_proportion <= 0.f) {
    return fit_knn_alignment_point_to_plane(X, Y, state, config.k,
                                            config.sigma);
  }

  using T = tf::coordinate_type<Policy0, Policy1>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(Dims == 3, "Point-to-plane alignment requires 3D points");
  static_assert(Dims == tf::coordinate_dims_v<Policy1>,
                "Point sets must have the same dimensionality");
  static_assert(tf::has_tree_policy<Policy1>,
                "Target point set Y must have a tree policy attached");
  static_assert(tf::has_normals_policy<Policy1>,
                "Target point set Y must have normals attached");

  const auto n = X.size();
  const auto k = config.k;
  const auto sigma = config.sigma;
  const auto &Y_normals = Y.normals();

  state.src_indices.allocate(n);
  state.target_points.allocate(n);
  state.target_normals.allocate(n);
  state.distances.allocate(n);

  // Initialize indices
  tf::parallel_iota(state.src_indices, 0);

  if (k == 1) {
    tf::parallel_for_each(
        tf::zip(X, state.target_points, state.target_normals, state.distances),
        [&](auto tup) {
          auto &&[x, tgt_out, n_out, dist_out] = tup;
          auto query = tf::transformed(x, tf::frame_of(X));
          auto [id, cpt] = tf::neighbor_search(Y, query);
          tgt_out = cpt.point;
          n_out = tf::transformed_normal(Y_normals[id], tf::frame_of(Y));
          dist_out = cpt.metric;
        },
        tf::checked);
  } else {
    tf::parallel_for_each(
        tf::zip(X, state.target_points, state.target_normals, state.distances),
        [&](auto tup) {
          constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
          auto &&[x, tgt_out, n_out, dist_out] = tup;
          std::array<
              tf::nearest_neighbor<typename Policy1::index_type, T, Dims>, 10>
              knn_buffer;
          auto knn = tf::make_nearest_neighbors(knn_buffer.begin(),
                                                std::min(k, std::size_t(10)));
          auto query = tf::transformed(x, tf::frame_of(X));
          tf::neighbor_search(Y, query, knn);

          auto sig = sigma < 0 ? knn.metric() : sigma * sigma;
          tf::point<T, Dims> weighted_pt = tf::zero;
          tf::vector<T, Dims> normal_sum = tf::zero;
          T w = 0;

          for (const auto &neighbor : knn) {
            auto l_w = std::exp(-neighbor.metric() / (T(2) * sig));
            w += l_w;
            weighted_pt += neighbor.info.point.as_vector_view() * l_w;
            normal_sum += Y_normals[neighbor.element] * l_w;
          }
          weighted_pt.as_vector_view() /= w;

          tgt_out = weighted_pt;
          n_out = tf::transformed_normal(tf::make_unit_vector(normal_sum),
                                         tf::frame_of(Y));
          dist_out = knn_buffer.front().metric();
        },
        tf::checked);
  }

  tbb::parallel_sort(
      state.src_indices.begin(), state.src_indices.end(),
      [&](int a, int b) { return state.distances[a] < state.distances[b]; });

  auto keep_n = n - std::size_t(n * config.outlier_proportion);
  auto kept_indices = tf::take(state.src_indices, keep_n);

  auto filtered_target =
      tf::make_points(
          tf::make_indirect_range(kept_indices, state.target_points.points())) |
      tf::tag_normals(tf::make_unit_vectors(tf::make_indirect_range(
          kept_indices, state.target_normals.unit_vectors())));

  if constexpr (tf::has_normals_policy<Policy0>) {
    auto filtered_source =
        tf::make_points(tf::make_indirect_range(kept_indices, X)) |
        tf::tag(tf::frame_of(X)) |
        tf::tag_normals(tf::make_unit_vectors(
            tf::make_indirect_range(kept_indices, X.normals())));
    return tf::geometry::fit_rigid_alignment_point_to_plane(
        filtered_source, filtered_target, state.alignment_state);
  } else {
    auto filtered_source =
        tf::make_points(tf::make_indirect_range(kept_indices, X)) |
        tf::tag(tf::frame_of(X));
    return tf::geometry::fit_rigid_alignment_point_to_plane(
        filtered_source, filtered_target, state.alignment_state);
  }
}

/// @brief Fit rigid transformation with outlier rejection (point-to-plane,
/// allocates internally).
template <typename Policy0, typename Policy1>
auto fit_knn_alignment_point_to_plane(const tf::points<Policy0> &X,
                                      const tf::points<Policy1> &Y,
                                      const tf::knn_alignment_config &config) {
  knn_alignment_plane_state<tf::coordinate_type<Policy1>,
                            tf::coordinate_dims_v<Policy1>>
      state;
  return fit_knn_alignment_point_to_plane(X, Y, state, config);
}

} // namespace tf::geometry
