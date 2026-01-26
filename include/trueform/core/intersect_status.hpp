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

/// @ingroup core_queries
/// @brief Enumeration of possible outcomes for intersection queries.
///
/// Used to indicate the status of intersection tests between geometric
/// primitives, such as lines, rays, segments, or polygons.
enum class intersect_status {
  /// No intersection occurred.
  none = 0,

  /// A valid intersection was found.
  intersection = 1,

  /// The objects are parallel
  parallel = 2,

  coplanar = 3,
  colinear = 3,
  non_parallel = 4
};

} // namespace tf
