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

#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "./edge.hpp"
#include "./vertex.hpp"

#include <algorithm>

namespace tf::intersect::graph {

/// Apply VV remap via union-find.
///
/// Deduplicates VV pairs, builds union-find, remaps edge endpoints
/// and loop vertices. Returns true if any VV pairs existed.
template <typename Index, typename VVRange>
auto apply_vv_remap(
    VVRange &&vv_range, Index remap_size,
    tf::offset_block_buffer<Index, edge<Index>> &edge_defs,
    tf::offset_block_buffer<Index, vertex<Index>> &loops,
    tf::buffer<Index> &point_remap) -> bool {
  auto vv_unique_end =
      std::unique(vv_range.begin(), vv_range.end(),
                  [](const auto &a, const auto &b) {
                    return a.point_a == b.point_a && a.point_b == b.point_b;
                  });
  if (vv_range.begin() == vv_unique_end)
    return false;

  point_remap.allocate(remap_size);
  for (Index i = 0; i < Index(remap_size); ++i)
    point_remap[i] = i;

  auto find_root = [&](Index x) -> Index {
    while (point_remap[x] != x)
      x = point_remap[x] = point_remap[point_remap[x]];
    return x;
  };

  for (auto it = vv_range.begin(); it != vv_unique_end; ++it) {
    auto a = find_root(it->point_a);
    auto b = find_root(it->point_b);
    if (a != b)
      point_remap[std::max(a, b)] = std::min(a, b);
  }
  for (Index i = 0; i < Index(remap_size); ++i)
    point_remap[i] = find_root(i);

  // Remap edge endpoints
  for (auto &&grp : edge_defs)
    for (auto &e : grp) {
      e.point_0 = point_remap[e.point_0];
      e.point_1 = point_remap[e.point_1];
    }

  // Remap loop vertices (improvement C)
  for (auto &&loop : loops)
    for (auto &v : loop)
      if (v.source == vertex_source::created)
        v.id = point_remap[v.id];

  return true;
}

} // namespace tf::intersect::graph
