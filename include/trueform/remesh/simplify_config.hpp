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
/// @brief Configuration for error-budget mesh simplification (tf::simplify).
///
/// Extends collapse_guard_config: every edge whose quadric error is within
/// error_rel * bounding-box-diagonal is collapsed, cheapest first, to
/// exhaustion, then a min-angle flip + tangential-relaxation cleanup runs.
/// Unlike decimate this has no target face count -- flat regions collapse for
/// ~0 error while curved detail and features survive.
///
/// The error-based counterpart of isotropic_remesh_config: with iterations > 1
/// it becomes an iterated remesh (collapse + cleanup, re-collapsing each round,
/// since the flip pass unblocks collapses the quality guard had stopped).
/// iterations = 1 is a single faithful collapse + cleanup. Defaults to
/// stabilizer = 1e-3 and check_normals = true (quadric collapse should not
/// invert faces), like decimate_config.
///
/// @tparam Real The scalar type.
template <typename Real>
struct simplify_config : collapse_guard_config<Real> {
  /// Error budget as a fraction of the mesh bounding-box diagonal. An edge is
  /// collapsed when its quadric error is <= error_rel * diagonal.
  Real error_rel = Real(0.002);

  /// Outer collapse rounds. 1 = a single collapse + cleanup (plain simplify).
  /// > 1 re-collapses after each cleanup -- the flip pass unblocks collapses
  /// the quality guard had stopped, so more is removed (an iterated remesh).
  /// Higher trades fidelity-to-original for coarseness.
  int iterations = 1;

  /// Quality cleanup after each collapse: this many rounds of min-angle edge
  /// flip + tangential relaxation (feature/region/boundary-aware). 0 = pure
  /// error-budget collapse with no flips. Default 3.
  int optimize_iterations = 3;

  /// Tangential relaxation passes per cleanup round.
  int relaxation_iters = 3;

  /// Damping factor for tangential relaxation in (0, 1].
  Real lambda = Real(0.5);

  simplify_config(Real error_rel = Real(0.002), Real min_quality = Real(0.3),
                  bool preserve_boundary = true, bool parallel = true,
                  tf::rad<Real> feature_angle = tf::rad<Real>(Real(-1)),
                  Real feature_weight = Real(100), double stabilizer = 1e-3,
                  bool check_normals = true, int optimize_iterations = 3,
                  int iterations = 1, int relaxation_iters = 3,
                  Real lambda = Real(0.5))
      : collapse_guard_config<Real>{min_quality, check_normals,
                                    preserve_boundary, true, parallel,
                                    feature_angle, feature_weight, stabilizer},
        error_rel(error_rel), iterations(iterations),
        optimize_iterations(optimize_iterations),
        relaxation_iters(relaxation_iters), lambda(lambda) {}
};

} // namespace tf
