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
/// @brief collapse_config plus the geometric acceptance guards shared by every
/// collapse-based operation (decimate, simplify, length collapse, isotropic and
/// error remesh): the [0,1] quality floor and the normal-inversion guard.
///
/// Both are consumed when an operation builds its collapse checker (and, in the
/// remeshers, to decide whether the edge-flip pass uses the fold guard). They
/// live here rather than in collapse_config because collapse_config is the
/// slice the collapse_handler stores; the guards belong to the checker.
///
/// @tparam Real The scalar type.
template <typename Real> struct collapse_guard_config : collapse_config<Real> {
  /// Worst triangle quality allowed after a collapse, in [0,1] (1 =
  /// equilateral, ->0 = sliver; quality = (2/sqrt3)*(2*area)/max_edge^2).
  ///   <0 -> disable the quality check.
  ///    0 -> never worsen the worst surrounding triangle.
  ///   >0 -> quality floor (still permits matching an already-worse ring, and
  ///         always permits an improvement).
  Real min_quality = Real(-1);

  /// Reject any collapse -- and, in the remeshers, any edge flip -- that would
  /// invert a triangle's normal (a fold). Runtime guard.
  bool check_normals = false;

  collapse_guard_config(Real min_quality = Real(-1),
                          bool check_normals = false,
                          bool preserve_boundary = true, bool use_quadric = true,
                          bool parallel = true,
                          tf::rad<Real> feature_angle = tf::rad<Real>(Real(-1)),
                          Real feature_weight = Real(100),
                          double stabilizer = 1e-6)
      : collapse_config<Real>{preserve_boundary, use_quadric, parallel,
                              feature_angle, feature_weight, stabilizer},
        min_quality(min_quality), check_normals(check_normals) {}
};

} // namespace tf
