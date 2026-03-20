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
/// @brief Shared configuration for collapse_handler.
///
/// Controls the machinery (quadrics, feature mask, boundary preservation,
/// parallelism) that is independent of the scoring/acceptance policy.
///
/// @tparam Real The scalar type.
template <typename Real> struct collapse_config {
  bool preserve_boundary = true;
  bool use_quadric = true;
  bool parallel = true;
  tf::rad<Real> feature_angle{Real(-1)};
  Real feature_weight = Real(100);
  double stabilizer = 1e-6;
};

} // namespace tf
