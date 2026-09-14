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

namespace tf {

/// @ingroup volume
/// @brief How @ref tf::isosurface extracts the surface.
///
/// `flying_edges` places every vertex on a grid edge: the fastest regular
/// output, defined for any scalar field, and the compatibility default.
/// `dual_contouring` places one vertex per surface component of a cell, fitted
/// to the field's own crossings, which recovers creases and corners a
/// grid-edge vertex cannot represent. It assumes a distance-like field.
enum class isosurface_method {
  flying_edges,
  dual_contouring,
};

/// @ingroup volume
/// @brief Parameters of an isosurface extraction.
///
/// Implicitly constructible from an @ref tf::isosurface_method, so a call site
/// that only chooses the method writes it directly.
struct isosurface_config {
  isosurface_config() = default;
  isosurface_config(isosurface_method m) : method(m) {}

  /// @brief Which extractor produces the surface.
  isosurface_method method = isosurface_method::flying_edges;

  /// @brief Dual contouring only: recover sharp features by refitting each
  /// feature vertex to the planes its neighbourhood's crossings state. One
  /// deterministic pass. Flying edges ignores it.
  bool refine = true;

  /// @brief Dual contouring only: the dimensionless weight of the pull toward
  /// the crossing centroid, `lambda = stabilizer * trace(A) / 3`. Must be
  /// finite and nonnegative; zero leaves an ill-conditioned cell unregularized.
  double stabilizer = 0.01;
};

} // namespace tf
