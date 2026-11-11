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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/cdt/dt3/tet_position.hpp"
#include "./tet_entrance_workspace.hpp"
#include <array>
#include <cstddef>
#include <tbb/parallel_sort.h>
#include <utility>

namespace tf::tetmesh {

/// THE ENTRANCE'S POINT IDENTITY: the points `points` in one cell of the
/// reach grid of side `quantum` (@ref tf::topology::cdt::dt3::tet_reach_cell)
/// are one point from the start, the first of them in the numbering. The
/// points are renumbered densely in their order, `rows` restated on the
/// survivors — a row two of whose corners became one is the surface
/// statement's to throw out (@ref merge_tet_facets) — and the count of
/// points that joined another is returned. The work walks on `work`. One
/// parallel sort by cell, then an adjacent sweep and a prefix, serial,
/// and the restatement in parallel: O(n log n + rows).
template <typename Index, typename Int>
auto state_tet_reach_cells(tf::points_buffer<Int, 3> &points,
                           tf::blocked_buffer<Index, 3> &rows, Int quantum,
                           tet_entrance_workspace<Index, Int> &work)
    -> std::size_t {
  namespace dt3 = tf::topology::cdt::dt3;
  const std::size_t n = points.size();
  auto &keys = work.cell_keys;
  keys.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&keys, &points, quantum](std::size_t i) {
        const auto &p = points[i];
        keys[i] = {{dt3::tet_reach_cell(Int(p[0]), quantum),
                    dt3::tet_reach_cell(Int(p[1]), quantum),
                    dt3::tet_reach_cell(Int(p[2]), quantum)},
                   Index(i)};
      },
      tf::checked);
  tbb::parallel_sort(keys.begin(), keys.end());
  auto &survivor = work.cell_point;
  survivor.allocate(n);
  std::size_t joined = 0;
  for (std::size_t t = 0; t < n; ++t) {
    const bool head = t == 0 || keys[t].cell != keys[t - 1].cell;
    survivor[std::size_t(keys[t].point)] =
        head ? keys[t].point : survivor[std::size_t(keys[t - 1].point)];
    joined += !head;
  }
  if (!joined)
    return 0;
  auto &dense = work.cell_dense;
  dense.allocate(n);
  Index next = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (survivor[i] == Index(i))
      dense[i] = next++;
  auto &kept = work.cell_points;
  kept.allocate(std::size_t(next));
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](std::size_t i) {
        if (survivor[i] == Index(i))
          kept[std::size_t(dense[i])] = points[i];
      },
      tf::checked);
  std::swap(points, kept);
  tf::parallel_for_each(
      rows,
      [&survivor, &dense](auto &&row) {
        for (auto &corner : row)
          corner = dense[std::size_t(survivor[std::size_t(corner)])];
      },
      tf::checked);
  return joined;
}

} // namespace tf::tetmesh
