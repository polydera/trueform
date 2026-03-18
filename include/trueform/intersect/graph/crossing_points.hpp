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

#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../exact/coplanar_edge_edge_point.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/vertex.hpp"
#include "./edge.hpp"

namespace tf::intersect::graph {

/// Compute EE crossing point coordinates.
///
/// Groups EE records by triple, computes one crossing point per unique triple.
///
/// @param ee_range         Sorted EE record range
/// @param edge_defs        Canonical edge groups
/// @param get_point        Point accessor
/// @param crossing_points  [out] Computed crossing point coordinates
/// @return EE group offsets (for collect_split_entries)
template <typename Index, typename EERange, typename GetPoint>
auto compute_ee_crossing_points(
    const EERange &ee_range,
    const tf::offset_block_buffer<Index, edge<Index>> &edge_defs,
    const GetPoint &get_point,
    tf::buffer<tf::point<int32_t, 3>> &crossing_points)
    -> tf::buffer<Index> {
  auto ee_count = ee_range.size();

  tf::buffer<Index> ee_offsets;
  if (ee_count == 0)
    return ee_offsets;

  ee_offsets.reserve(ee_count + 1);
  tf::compute_offsets(ee_range, std::back_inserter(ee_offsets), Index(0),
                      [](const auto &a, const auto &b) {
                        return a.triple == b.triple;
                      });
  auto num_unique = ee_offsets.size() - 1;

  crossing_points.allocate(num_unique);

  auto ee_groups = tf::make_offset_block_range(ee_offsets, ee_range);
  tf::parallel_for_each(tf::enumerate(ee_groups), [&](auto pair) {
    auto &&[g, group] = pair;
    auto &r = group[0];
    auto &&ea = edge_defs[r.edge_a][0];
    auto &&eb = edge_defs[r.edge_b][0];
    auto pa0 = get_point(-1, ea.point_0);
    auto pa1 = get_point(-1, ea.point_1);
    auto pb0 = get_point(-1, eb.point_0);
    auto pb1 = get_point(-1, eb.point_1);
    auto [ax0, ax1] = tf::exact::projection_axes(pa0, pa1, pb0);
    tf::exact::vertex va0{ea.point_0, pa0}, va1{ea.point_1, pa1};
    tf::exact::vertex vb0{eb.point_0, pb0}, vb1{eb.point_1, pb1};
    crossing_points[g] =
        tf::exact::coplanar_edge_edge_point(va0, va1, vb0, vb1, ax0, ax1);
  });

  return ee_offsets;
}

} // namespace tf::intersect::graph
