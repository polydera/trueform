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

#include "../core/policy/normals.hpp"
#include "./impl/fit_knn_alignment_point_to_plane.hpp"
#include "./impl/fit_knn_alignment_point_to_point.hpp"
#include "./knn_alignment_config.hpp"
#include "./knn_alignment_state.hpp"

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
/// If Y has normals attached (via `tf::tag_normals`), uses point-to-plane
/// error metric which converges faster in ICP loops. Otherwise uses
/// point-to-point.
///
/// This is equivalent to one iteration of ICP when k=1. For k>1, soft
/// correspondences provide robustness to noise and partial overlap.
///
/// When `config.outlier_proportion > 0`, the worst correspondences are
/// rejected before fitting, providing robustness to partial overlap.
///
/// @param X Source point set.
/// @param Y Target point set with tree (searched for neighbors).
/// @param state Reusable workspace.
/// @param config Configuration (k, sigma, outlier_proportion).
/// @return Rigid transform mapping X -> Y.
///
/// @see @ref tf::knn_alignment_config
/// @see @ref tf::knn_alignment_state
template <typename Policy0, typename Policy1>
auto fit_knn_alignment(const tf::points<Policy0> &X,
                       const tf::points<Policy1> &Y,
                       knn_alignment_state<Policy0, Policy1> &state,
                       const tf::knn_alignment_config &config = {})
    -> tf::transformation<tf::coordinate_type<Policy0, Policy1>,
                          tf::coordinate_dims_v<Policy0>> {
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;

  if constexpr (tf::has_normals_policy<Policy1> && Dims == 3) {
    return geometry::fit_knn_alignment_point_to_plane(X, Y, state, config);
  } else {
    return geometry::fit_knn_alignment_point_to_point(X, Y, state, config);
  }
}

/// @ingroup geometry_registration
/// @brief Fit a rigid transformation using k-nearest neighbor correspondences
/// (allocates internally).
///
/// @see @ref fit_knn_alignment(const tf::points<Policy0>&, const
/// tf::points<Policy1>&, knn_alignment_state<Policy0, Policy1>&, const
/// tf::knn_alignment_config&)
template <typename Policy0, typename Policy1>
auto fit_knn_alignment(const tf::points<Policy0> &X,
                       const tf::points<Policy1> &Y,
                       const tf::knn_alignment_config &config = {})
    -> tf::transformation<tf::coordinate_type<Policy0, Policy1>,
                          tf::coordinate_dims_v<Policy0>> {
  knn_alignment_state<Policy0, Policy1> state;
  return fit_knn_alignment(X, Y, state, config);
}

} // namespace tf
