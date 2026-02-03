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
/// @param X Source point set.
/// @param Y Target point set with tree (searched for neighbors).
/// @param k Number of nearest neighbors (default: 1 = classic ICP).
/// @param sigma Gaussian kernel width. If negative, uses the k-th neighbor
///              distance as sigma (adaptive). Default: -1.
/// @return Rigid transform mapping X -> Y.
template <typename Policy0, typename Policy1>
auto fit_knn_alignment(const tf::points<Policy0> &X,
                       const tf::points<Policy1> &Y, std::size_t k = 1,
                       tf::coordinate_type<Policy0, Policy1> sigma = -1)
    -> tf::transformation<tf::coordinate_type<Policy0, Policy1>,
                          tf::coordinate_dims_v<Policy0>> {
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;

  if constexpr (tf::has_normals_policy<Policy1> && Dims == 3) {
    return geometry::fit_knn_alignment_point_to_plane(X, Y, k, sigma);
  } else {
    return geometry::fit_knn_alignment_point_to_point(X, Y, k, sigma);
  }
}

} // namespace tf
