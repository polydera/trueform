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

/// Name for the self-only build of @ref tf::polygon_intersections:
/// `build(form)` is self-only by definition there — the alias and the
/// class have the same one-form contract.
template <typename Index, typename RealType, typename Int = tf::exact::int32>
using intersections_within_polygons =
    tf::polygon_intersections<Index, RealType, Int>;

} // namespace tf
