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
#include "../exact/make_grid_points.hpp"

namespace tf::clean {

/// Invokes `f(view)` with either the input points (when `tolerance == 0`)
/// or a snapped view via `tf::make_grid_points`.
template <typename Policy, typename F>
auto with_snapped_points(const tf::points<Policy> &points,
                         tf::coordinate_type<Policy> tolerance, F &&f) {
  if (tolerance == 0)
    return f(points);
  return f(tf::make_grid_points(points, tolerance));
}

} // namespace tf::clean
