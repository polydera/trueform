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

#include "../core/angle.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Configuration for quadric error metric decimation.
///
/// @tparam Real The scalar type.
template <typename Real> struct decimate_config {
  /// Maximum aspect ratio allowed after a collapse.
  /// Set negative to disable the check.
  Real max_aspect_ratio = 40;

  /// If true, boundary edges are never collapsed.
  bool preserve_boundary = false;

  /// Tikhonov stabilizer for quadric solve.
  double stabilizer = 1e-3;

  /// If true, use parallel partitioned collapse. If false, sequential.
  bool parallel = true;

  /// Dihedral angle threshold for feature edge detection.
  /// Edges sharper than this are preserved. Negative disables.
  tf::rad<Real> feature_angle{Real(-1)};

  /// Penalty weight for feature edge quadrics, relative to mean face
  /// quadric trace. Higher values preserve features more rigidly.
  Real feature_weight = Real(100);
};

} // namespace tf
