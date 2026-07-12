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

/// @ingroup topology
/// @brief Configuration for @ref tf::cdt_refiner.
struct cdt_refine_config {
  /// Quality floor for triangles, q = (2/sqrt3) * 2A / max_edge^2 in (0, 1]
  /// (1 = equilateral) -- the same measure as
  /// collapse_guard_config::min_quality. Values above 0.45 are clamped:
  /// circumcenter refinement is not guaranteed to terminate beyond it.
  float min_quality = 0.3f;

  /// Permit midpoint splits of encroached constraint edges. When false the
  /// constraint polylines are frozen and triangles pinned against them keep
  /// whatever quality the input allows.
  bool split_encroached = true;
};

} // namespace tf
