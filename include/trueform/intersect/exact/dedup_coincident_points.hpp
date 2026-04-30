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

#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/index_map.hpp"
#include "../../core/points.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../exact/vertex.hpp"
#include "./tagged_intersection.hpp"

namespace tf::intersect {

/// Identify all intersection points with identical int coordinates.
/// Points that share the same exact coordinate get the same ID.
template <typename Index, typename Int>
void dedup_coincident_points(
    tf::buffer<tagged_intersection<Index>> &intersections,
    tf::buffer<tf::exact::pt3<Int>> &points) {
  if (points.size() == 0)
    return;
  tf::index_map_buffer<Index> im;
  tf::make_unique_index_map(tf::make_points(points), im);
  if (im.kept_ids().size() == points.size())
    return; // no duplicates
  // Compact points
  tf::buffer<tf::exact::pt3<Int>> new_pts;
  new_pts.allocate(im.kept_ids().size());
  tf::parallel_copy(tf::make_indirect_range(im.kept_ids(), points), new_pts);
  // Remap record IDs
  tf::parallel_for_each(intersections, [&](tagged_intersection<Index> &rec) {
    rec.id = im.f()[rec.id];
  });
  points = std::move(new_pts);
}

} // namespace tf::intersect
