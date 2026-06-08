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

#include <limits>

namespace tf {

/// @ingroup remesh
/// @brief Configuration for edge-length-based collapse.
///
/// Extends collapse_guard_config (min_quality + check_normals) with a
/// post-collapse length cap, for the convenience API (collapse_short_edges).
///
/// @tparam Real The scalar type.
template <typename Real>
struct length_collapse_config : collapse_guard_config<Real> {
  Real max_length = std::numeric_limits<Real>::max();

  length_collapse_config(
      Real max_length = std::numeric_limits<Real>::max(),
      Real min_quality = Real(-1), bool preserve_boundary = true,
      bool use_quadric = true, bool parallel = true,
      tf::rad<Real> feature_angle = tf::rad<Real>(Real(-1)),
      Real feature_weight = Real(100), double stabilizer = 1e-6,
      bool check_normals = false)
      : collapse_guard_config<Real>{min_quality, check_normals,
                                    preserve_boundary, use_quadric, parallel,
                                    feature_angle, feature_weight, stabilizer},
        max_length(max_length) {}
};

} // namespace tf
