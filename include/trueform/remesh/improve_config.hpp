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

#include <cstdint>

namespace tf {

/// @ingroup remesh
/// @brief What an edge flip optimizes during improve_triangulation.
///   - valence:   flip toward the ideal vertex valence (6 interior, 4 boundary).
///   - min_angle: flip when it raises the smaller opposite angle (Delaunay-style),
///                which clears slivers a valence flip cannot.
enum class flip_objective : std::uint8_t { valence, min_angle };

/// @ingroup remesh
/// @brief Configuration for improve_triangulation (the fixed-count quality pass:
/// edge flips + tangential relaxation).
///
/// @tparam Real The scalar type.
template <typename Real> struct improve_config {
  /// Number of {flip, then relax} rounds. Interleaving lets a flip benefit from
  /// the previous round's relaxation.
  int iterations = 3;

  /// Tangential relaxation passes per round.
  int relaxation_iters = 3;

  /// Damping factor for tangential relaxation in (0, 1].
  Real lambda = Real(0.5);

  /// Reject any flip that would invert a triangle's normal (a fold). Applies to
  /// the valence objective; the min_angle objective always fold-guards.
  bool check_normals = true;

  /// What the flip pass optimizes.
  flip_objective flip = flip_objective::valence;
};

} // namespace tf
