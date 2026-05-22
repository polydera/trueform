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

#include "../core/coordinate_type.hpp"
#include "../core/points.hpp"
#include "../core/views/mapped_range.hpp"
#include "./snap.hpp"

namespace tf {

/// @ingroup exact_points
/// @brief View of `points` snapped to a grid of cell width `tol + 1`
///        (integer coords) or `tol` (float coords). Same coord type as input.
template <typename Policy>
auto make_grid_points(const tf::points<Policy> &points,
                      tf::coordinate_type<Policy> tol) {
  return tf::make_points(tf::make_mapped_range(
      points, [tol](const auto &p) { return tf::exact::snap_point(p, tol); }));
}

} // namespace tf
