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
/// @brief Configuration for quadric error metric decimation.
///
/// Extends collapse_guard_config (min_quality + check_normals) for the
/// convenience API (decimate). Defaults to stabilizer = 1e-3 and
/// check_normals = true (quadric decimation should not invert faces).
///
/// @tparam Real The scalar type.
template <typename Real>
struct decimate_config : collapse_guard_config<Real> {
  decimate_config(Real min_quality = Real(-1), bool preserve_boundary = true,
                  bool parallel = true,
                  tf::rad<Real> feature_angle = tf::rad<Real>(Real(-1)),
                  Real feature_weight = Real(100), double stabilizer = 1e-3,
                  bool check_normals = true)
      : collapse_guard_config<Real>{min_quality, check_normals,
                                    preserve_boundary, true, parallel,
                                    feature_angle, feature_weight,
                                    stabilizer} {}
};

} // namespace tf
