/*
 * Copyright (c) 2026 XLAB
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

#include "../intersect/intersect_config.hpp"
#include "../topology/triangulation_type.hpp"

namespace tf {

/// @ingroup cut
/// @brief Configuration for an arrangement build: the intersection run
///        plus the triangulation built over the cut surfaces.
///
/// Implicitly constructible from @ref tf::intersect_config,
/// @ref tf::intersect_mode, or @ref tf::triangulation_type alone, so
/// call sites spelling only the part they care about keep working.
/// The default intersect mode is the arrangement surfaces' default
/// (`primitives | resolve_crossing_contours`), not
/// `intersect_config`'s bare `primitives`.
struct arrangement_config {
  intersect_config intersect = {intersect_mode::primitives |
                                intersect_mode::resolve_crossing_contours};
  triangulation_type triangulation = triangulation_type::cdt;

  constexpr arrangement_config() = default;
  constexpr arrangement_config(intersect_config c) : intersect(c) {}
  constexpr arrangement_config(intersect_mode m) : intersect(m) {}
  constexpr arrangement_config(intersect_mode m, double tolerance)
      : intersect(m, tolerance) {}
  constexpr arrangement_config(triangulation_type t) : triangulation(t) {}
  constexpr arrangement_config(intersect_config c, triangulation_type t)
      : intersect(c), triangulation(t) {}
};

} // namespace tf
