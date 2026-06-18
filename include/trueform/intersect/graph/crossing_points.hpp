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

#include "../../core/algorithm/parallel_for_each.hpp"
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
/// One point per triple group: a junction is observed once per containing
/// face as a different edge pair, and the triple is the cross-face
/// invariant. A group with >= 2 distinct coplanar-stamped edges is a
/// contact pack, where the triple is ambiguous: it refines by edge pair.
///
/// @param ee_range         Sorted EE record range (triple, then edge pair)
/// @param edge_defs        Canonical edge groups
/// @param get_point        Point accessor
/// @param crossing_points  [out] Computed crossing point coordinates (Int precision)
/// @return EE group offsets (for collect_split_entries)
template <typename Index, typename EERange, typename GetPoint, typename Int>
auto compute_ee_crossing_points(
    const EERange &ee_range,
    const tf::offset_block_buffer<Index, edge<Index>> &edge_defs,
    const GetPoint &get_point, tf::buffer<tf::point<Int, 3>> &crossing_points)
    -> tf::buffer<Index> {
  auto ee_count = static_cast<Index>(ee_range.size());

  tf::buffer<Index> ee_offsets;
  if (ee_count == 0)
    return ee_offsets;

  auto rec = ee_range.begin();

  auto is_stamped = [&](Index ce) {
    for (const auto &inst : edge_defs[ce])
      if (inst.from_coplanar)
        return true;
    return false;
  };

  // Two distinct stamped canonical edges in one triple group = a face pair
  // meeting in more than one contact edge: the triple under-determines the
  // crossing there.
  auto is_pack = [&](Index begin, Index end) {
    Index first = -1;
    auto check = [&](Index ce) {
      if (!is_stamped(ce))
        return false;
      if (first == -1) {
        first = ce;
        return false;
      }
      return ce != first;
    };
    for (auto i = begin; i != end; ++i)
      if (check(rec[i].edge_a) || check(rec[i].edge_b))
        return true;
    return false;
  };

  ee_offsets.reserve(ee_count + 1);
  ee_offsets.push_back(0);
  for (Index begin = 0; begin != ee_count;) {
    Index end = begin + 1;
    while (end != ee_count && rec[end].triple == rec[begin].triple)
      ++end;
    if (is_pack(begin, end))
      for (auto i = begin + 1; i != end; ++i) {
        if (rec[i].edge_a != rec[i - 1].edge_a ||
            rec[i].edge_b != rec[i - 1].edge_b)
          ee_offsets.push_back(i);
      }
    ee_offsets.push_back(end);
    begin = end;
  }
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
    // Axes from BOTH edge directions, not 3 points: pb0 may lie on edge_a at a
    // T-junction (triple point), which makes the 3-point normal degenerate and
    // throws the crossing far off the edge. (a1-a0)x(b1-b0) is robust there.
    auto [ax0, ax1] = tf::exact::projection_axes_edges(pa0, pa1, pb0, pb1);
    tf::exact::vertex<Index, Int> va0{ea.point_0, pa0}, va1{ea.point_1, pa1};
    tf::exact::vertex<Index, Int> vb0{eb.point_0, pb0}, vb1{eb.point_1, pb1};
    crossing_points[g] =
        tf::exact::coplanar_edge_edge_point(va0, va1, vb0, vb1, ax0, ax1);
  });

  return ee_offsets;
}

} // namespace tf::intersect::graph
