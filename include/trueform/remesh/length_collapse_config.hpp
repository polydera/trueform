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

#include <limits>

namespace tf {

/// @ingroup remesh
/// @brief Configuration for edge-length-based collapse.
///
/// Shared parameters for both exhaustion-mode and target-mode
/// short-edge collapse.
///
/// @tparam Real The scalar type.
template <typename Real> struct length_collapse_config {
  /// Maximum edge length allowed after a collapse. Collapses that
  /// would create longer edges are rejected.
  Real max_length = std::numeric_limits<Real>::max();

  /// If true, boundary edges are never collapsed.
  bool preserve_boundary = false;

  /// If true, use quadric error metric for vertex placement.
  /// If false, keep the surviving vertex at its original position.
  bool use_quadric = true;

  /// Maximum aspect ratio allowed after a collapse.
  Real max_aspect_ratio = 40;

  /// Tikhonov stabilizer for quadric solve.
  double stabilizer = 1e-6;

  /// If true, use parallel partitioned collapse. If false, sequential.
  bool parallel = true;
};

} // namespace tf
