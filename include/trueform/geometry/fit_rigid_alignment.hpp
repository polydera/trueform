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
#include "./impl/fit_rigid_alignment_point_to_plane.hpp"
#include "./impl/fit_rigid_alignment_point_to_point.hpp"

namespace tf {

/// @ingroup geometry_registration
/// @brief Fit a rigid transformation (rotation + translation) between two
/// corresponding point sets.
///
/// Computes the optimal rigid transformation T such that T(X) ≈ Y.
///
/// If Y has normals attached (via `tf::tag_normals`), uses point-to-plane
/// minimization which converges faster in ICP loops. Otherwise uses
/// point-to-point (Kabsch/SVD) which is optimal for exact correspondences.
///
/// If the point sets have frames attached, the alignment is computed
/// in world space (i.e., with frames applied).
///
/// @tparam Policy0 The policy type for the source point set.
/// @tparam Policy1 The policy type for the target point set.
/// @param X The source point set.
/// @param Y The target point set (must have same size as X).
/// @return A transformation that best aligns X to Y.
template <typename Policy0, typename Policy1>
auto fit_rigid_alignment(const tf::points<Policy0> &X,
                         const tf::points<Policy1> &Y)
    -> tf::transformation<tf::coordinate_type<Policy0, Policy1>,
                          tf::coordinate_dims_v<Policy0>> {
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;

  if constexpr (tf::has_normals_policy<Policy1> && Dims == 3) {
    return geometry::fit_rigid_alignment_point_to_plane(X, Y);
  } else {
    return geometry::fit_rigid_alignment_point_to_point(X, Y);
  }
}

} // namespace tf
