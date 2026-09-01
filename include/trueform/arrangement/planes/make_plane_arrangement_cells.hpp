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
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./plane_arrangement.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// The arrangement's cells, numbered densely across its planes.
///
/// The cell is the carrier: a pooled plane is ONE thing, so a stack's members
/// share the cells they cover and a coplanar duplicate carries its survivor's
/// cell. A cell no member covers emits no triangle and takes no number.
template <typename Index> struct plane_arrangement_cells {
  Index n_cells = 0;
  tf::buffer<Index> cell_of; // per triangle
};

/// Number the cells emission recorded. The plane owns its cells outright —
/// they meet other planes only at pieces — so the plane is the independent
/// grain and one prefix over the planes is the whole numbering.
///
/// Requires `record_triangle_cells()` before the build. A store the build
/// did not fill (or a weld compaction emptied) yields the empty numbering —
/// zero cells — never a stale read.
template <typename Index, typename Int>
auto make_plane_arrangement_cells(
    const plane_arrangement<Index, Int> &arrangement)
    -> plane_arrangement_cells<Index> {
  // emission states a plane-local cell per triangle it appends, never
  // negative, so the recorded id is directly the dense map's key
  const auto cells = arrangement.triangle_cells();
  const auto n_planes = arrangement.n_planes();

  plane_arrangement_cells<Index> out;
  if (cells.size() != arrangement.triangles().size())
    return out;
  out.cell_of.allocate(cells.size());
  tf::buffer<Index> plane_base;
  plane_base.allocate(std::size_t(n_planes) + 1);
  plane_base[0] = 0;

  struct local_t {
    tf::buffer<Index> dense;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index plane, local_t &local) {
        const auto range = arrangement.plane_range(plane);
        Index extent = 0;
        for (auto t = range[0]; t < range[1]; ++t)
          extent = std::max(extent, cells[std::size_t(t)] + Index(1));
        local.dense.allocate(std::size_t(extent));
        std::fill(local.dense.begin(), local.dense.end(), Index(-1));
        Index next = 0;
        for (auto t = range[0]; t < range[1]; ++t) {
          auto &dense = local.dense[std::size_t(cells[std::size_t(t)])];
          if (dense == Index(-1))
            dense = next++;
          out.cell_of[std::size_t(t)] = dense;
        }
        plane_base[std::size_t(plane) + 1] = next;
      },
      local_t{});

  for (std::size_t plane = 1; plane < plane_base.size(); ++plane)
    plane_base[plane] += plane_base[plane - 1];
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index plane) {
        const auto range = arrangement.plane_range(plane);
        const auto base = plane_base[std::size_t(plane)];
        for (auto t = range[0]; t < range[1]; ++t)
          out.cell_of[std::size_t(t)] += base;
      });
  out.n_cells = plane_base[std::size_t(n_planes)];
  return out;
}

} // namespace tf::arrangement
