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

#include "./collapse_guard_config.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Configuration for isotropic remeshing.
///
/// Extends collapse_guard_config (min_quality + check_normals). When
/// check_normals is set, the edge-flip pass uses the geometric fold guard.
///
/// @tparam Real The scalar type.
template <typename Real>
struct isotropic_remesh_config : collapse_guard_config<Real> {
  /// Target edge length.
  Real target_length;

  /// Number of outer iterations (split + collapse + flip + relax).
  int iterations = 3;

  /// Number of tangential relaxation iterations per outer iteration.
  int relaxation_iters = 3;

  /// Damping factor for tangential relaxation in (0, 1].
  Real lambda = Real(0.5);

  isotropic_remesh_config(Real target_length, int iterations = 3,
                int relaxation_iters = 3, Real min_quality = Real(0.3),
                Real lambda = Real(0.5), bool preserve_boundary = true,
                bool use_quadric = false, bool parallel = true,
                tf::rad<Real> feature_angle = tf::rad<Real>(Real(-1)),
                Real feature_weight = Real(100), double stabilizer = 1e-6,
                bool check_normals = false)
      : collapse_guard_config<Real>{min_quality, check_normals,
                                    preserve_boundary, use_quadric, parallel,
                                    feature_angle, feature_weight, stabilizer},
        target_length(target_length), iterations(iterations),
        relaxation_iters(relaxation_iters), lambda(lambda) {}
};

/// @brief Create a remesh config with just a target edge length.
template <typename Real>
auto make_isotropic_remesh_config(Real target_length) -> isotropic_remesh_config<Real> {
  return {target_length};
}

} // namespace tf
