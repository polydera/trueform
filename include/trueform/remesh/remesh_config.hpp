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

namespace tf {

/// @ingroup remesh
/// @brief Configuration for isotropic remeshing.
///
/// @tparam Real The scalar type.
template <typename Real> struct remesh_config {
  /// Target edge length. Edges longer than this are split,
  /// edges shorter are collapsed.
  Real target_length;

  /// Number of outer iterations (split + collapse + flip + relax).
  int iterations = 3;

  /// Number of tangential relaxation iterations per outer iteration.
  int relaxation_iters = 3;

  /// Maximum aspect ratio allowed after a collapse. Collapses that
  /// would create triangles with worse aspect ratio are rejected.
  /// Set negative to disable the check.
  Real max_aspect_ratio = -1;

  /// Damping factor for tangential relaxation in (0, 1].
  Real lambda = Real(0.5);

  /// If true, boundary edges are never split or collapsed.
  bool preserve_boundary = false;

  /// If true, use quadric error metric for collapse vertex placement.
  /// If false, keep the surviving vertex at its original position.
  bool use_quadric = false;

  /// If true, use parallel partitioned collapse. If false, sequential.
  bool parallel = true;

};

/// @brief Create a remesh config with just a target edge length.
template <typename Real>
auto make_remesh_config(Real target_length) -> remesh_config<Real> {
  return {target_length};
}

} // namespace tf
