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

#include "./collapse_config.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Configuration for isotropic remeshing.
///
/// @tparam Real The scalar type.
template <typename Real> struct remesh_config : collapse_config<Real> {
  /// Target edge length.
  Real target_length;

  /// Number of outer iterations (split + collapse + flip + relax).
  int iterations = 3;

  /// Number of tangential relaxation iterations per outer iteration.
  int relaxation_iters = 3;

  /// Maximum aspect ratio allowed after a collapse.
  /// Set negative to disable the check.
  Real max_aspect_ratio = Real(-1);

  /// Damping factor for tangential relaxation in (0, 1].
  Real lambda = Real(0.5);

  remesh_config(Real target_length, int iterations = 3,
                int relaxation_iters = 3, Real max_aspect_ratio = Real(-1),
                Real lambda = Real(0.5), bool preserve_boundary = true,
                bool use_quadric = false, bool parallel = true,
                tf::rad<Real> feature_angle = tf::rad<Real>(Real(-1)),
                Real feature_weight = Real(100), double stabilizer = 1e-6)
      : collapse_config<Real>{preserve_boundary, use_quadric, parallel,
                              feature_angle, feature_weight, stabilizer},
        target_length(target_length), iterations(iterations),
        relaxation_iters(relaxation_iters), max_aspect_ratio(max_aspect_ratio),
        lambda(lambda) {}
};

/// @brief Create a remesh config with just a target edge length.
template <typename Real>
auto make_remesh_config(Real target_length) -> remesh_config<Real> {
  return {target_length};
}

} // namespace tf
