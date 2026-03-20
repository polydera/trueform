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
/// @brief Configuration for quadric error metric decimation.
///
/// Extends collapse_config with geometric constraints for the
/// convenience API (decimate). Defaults to stabilizer = 1e-3.
///
/// @tparam Real The scalar type.
template <typename Real>
struct decimate_config : collapse_config<Real> {
  Real max_aspect_ratio = Real(40);

  decimate_config(Real max_aspect_ratio = 40, bool preserve_boundary = true,
                  bool parallel = true,
                  tf::rad<Real> feature_angle = tf::rad<Real>(Real(-1)),
                  Real feature_weight = Real(100), double stabilizer = 1e-3)
      : collapse_config<Real>{preserve_boundary, true, parallel, feature_angle,
                              feature_weight, stabilizer},
        max_aspect_ratio(max_aspect_ratio) {}
};

} // namespace tf
