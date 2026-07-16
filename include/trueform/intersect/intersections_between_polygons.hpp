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
#include "./polygon_intersections.hpp"

namespace tf {

/// Name for the pairwise builds of @ref tf::polygon_intersections:
/// without @ref tf::intersect_mode::within in the mode, the two-form
/// and N-range builds are cross-pair only. A mode carrying
/// @ref tf::intersect_mode::self_intersections adds per-form self
/// records through this name as well.
template <typename Index, typename RealType, typename Int = tf::exact::int32>
using intersections_between_polygons =
    tf::polygon_intersections<Index, RealType, Int>;

} // namespace tf
