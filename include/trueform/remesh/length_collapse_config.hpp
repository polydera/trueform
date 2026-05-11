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

#include <limits>

namespace tf {

/// @ingroup remesh
/// @brief Configuration for edge-length-based collapse.
///
/// Extends collapse_config with geometric constraints for the
/// convenience API (collapse_short_edges).
///
/// @tparam Real The scalar type.
template <typename Real>
struct length_collapse_config : collapse_config<Real> {
  Real max_length = std::numeric_limits<Real>::max();
  Real max_aspect_ratio = Real(-1);

  length_collapse_config(
      Real max_length = std::numeric_limits<Real>::max(),
      Real max_aspect_ratio = Real(-1), bool preserve_boundary = true,
      bool use_quadric = true, bool parallel = true,
      tf::rad<Real> feature_angle = tf::rad<Real>(Real(-1)),
      Real feature_weight = Real(100), double stabilizer = 1e-6)
      : collapse_config<Real>{preserve_boundary, use_quadric, parallel,
                              feature_angle, feature_weight, stabilizer},
        max_length(max_length), max_aspect_ratio(max_aspect_ratio) {}
};

} // namespace tf
