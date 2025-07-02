/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {

/// @ingroup geometry
/// @brief Enumeration of possible outcomes for intersection queries.
///
/// Used to indicate the status of intersection tests between geometric primitives,
/// such as lines, rays, segments, or polygons.
enum class intersect_status {
  /// No intersection occurred.
  none = 0,

  /// A valid intersection was found.
  intersection = 1,

  /// The objects are parallel
  parallel = 2,

  /// An error occurred during computation (e.g., degenerate input).
  error = 3
};

}
