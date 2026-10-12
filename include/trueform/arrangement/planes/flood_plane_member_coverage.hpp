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

#include "../../core/buffer.hpp"
#include "./plane_member_statements.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// Which of a pooled plane's members cover which of its regions, and with
/// which winding.
///
/// The triangulation states the regions — its components, region 0 the
/// hull exterior — so the only fact left is a winding: a member's own
/// side is a closed cycle of walls, and every wall belongs to exactly
/// the one member whose instance stated it, so crossing it moves that
/// member's winding alone. One flood from the exterior, where nobody winds
/// anything, settles all of them. A wall moves a winding by the multiplicity
/// the member states it with, signed by the side it is crossed from:
/// `cdt_orientation` is the triangulation's own winding in its projection.
///
/// A region's winding table is therefore its parent's moved by the wall
/// between them, and both are ascending runs of members, so the step is one
/// merge and the whole table costs what its facts cost. A member covers a
/// region where its winding is odd; `cover_range` is the dense side: per
/// region a ticket into `cover` and the aligned `cover_winding`, the
/// exterior's empty. Returns n_regions, or -1 for a carrier whose regions do
/// not all reach the exterior — a winding is only defined along a path from
/// it.
template <typename Index, typename Cdt>
auto flood_plane_member_coverage(
    const Cdt &cdt, int cdt_orientation,
    const tf::buffer<std::array<Index, 2>> &cons_row,
    const tf::buffer<plane_member_statement<Index>> &cons_statements,
    Index n_constraints, tf::buffer<std::array<Index, 4>> &adjacency,
    tf::buffer<Index> &offsets, tf::buffer<Index> &stack,
    tf::buffer<char> &reached, tf::buffer<std::array<Index, 2>> &cover_range,
    tf::buffer<Index> &cover, tf::buffer<signed char> &cover_winding)
    -> Index {
  const auto labels = cdt.region_labels();
  Index n_regions = 1;
  for (const auto label : labels)
    if (label >= n_regions)
      n_regions = label + Index(1);

  // a record's last lane is the side its region stands on: +1 when the
  // region's triangle runs the wall in the wall's own direction
  adjacency.clear();
  cdt.for_each_face_adjacency([&](Index tri, Index, Index, Index, Index label,
                                  const std::array<Index, 3> &neighbors,
                                  const std::array<bool, 3> &) {
    const auto owners = cdt.face_constraint_owners(tri);
    for (int e = 0; e < 3; ++e) {
      const auto neighbor = neighbors[std::size_t(e)];
      const auto other =
          neighbor == Index(-1) ? Index(0) : labels[std::size_t(neighbor)];
      if (other == label)
        continue;
      const auto &owner = owners[std::size_t(e)];
      const auto wall =
          owner.input_id == Index(-1) ? n_constraints : owner.input_id;
      const auto along = owner.t0 < owner.t1 ? Index(1) : Index(-1);
      adjacency.push_back({label, other, wall, along});
      if (neighbor == Index(-1))
        adjacency.push_back({other, label, wall, Index(-along)});
    }
  });
  std::sort(adjacency.begin(), adjacency.end());

  offsets.allocate(std::size_t(n_regions) + 1);
  std::fill(offsets.begin(), offsets.end(), Index(0));
  for (const auto &record : adjacency)
    ++offsets[std::size_t(record[0]) + 1];
  for (std::size_t r = 1; r < offsets.size(); ++r)
    offsets[r] += offsets[r - 1];

  cover_range.allocate(std::size_t(n_regions));
  cover.clear();
  cover_winding.clear();
  cover_range[0] = {Index(0), Index(0)};
  reached.allocate(std::size_t(n_regions));
  std::fill(reached.begin(), reached.end(), char(0));
  stack.clear();
  reached[0] = 1;
  stack.push_back(Index(0));
  while (stack.size() != 0) {
    const auto region = stack[stack.size() - 1];
    stack.pop_back();
    for (auto i = offsets[std::size_t(region)];
         i < offsets[std::size_t(region) + 1]; ++i) {
      const auto &record = adjacency[std::size_t(i)];
      const auto other = std::size_t(record[1]);
      if (reached[other])
        continue;
      reached[other] = 1;
      const auto parent = cover_range[std::size_t(region)];
      const auto wall = cons_row[std::size_t(record[2])];
      // crossing a wall the region stands left of moves the winding down
      const int sense = -int(record[3]) * cdt_orientation;
      const auto begin = Index(cover.size());
      auto held = parent[0];
      auto moved = wall[0];
      for (;;) {
        while (moved != wall[1] &&
               cons_statements[std::size_t(moved)].winding == 0)
          ++moved;
        if (held == parent[1] && moved == wall[1])
          break;
        if (moved == wall[1] ||
            (held != parent[1] &&
             cover[std::size_t(held)] <
                 cons_statements[std::size_t(moved)].member)) {
          // the parent's run lives in the buffer this appends to, so the
          // values are taken before an append may move it
          const auto member = cover[std::size_t(held)];
          const auto winding = cover_winding[std::size_t(held)];
          cover.push_back(member);
          cover_winding.push_back(winding);
          ++held;
          continue;
        }
        const auto &stated = cons_statements[std::size_t(moved)];
        const int step = sense * int(stated.winding);
        if (held != parent[1] && cover[std::size_t(held)] == stated.member) {
          const int winding = int(cover_winding[std::size_t(held)]) + step;
          if (winding != 0) {
            cover.push_back(stated.member);
            cover_winding.push_back(static_cast<signed char>(winding));
          }
          ++held;
          ++moved;
          continue;
        }
        cover.push_back(stated.member);
        cover_winding.push_back(static_cast<signed char>(step));
        ++moved;
      }
      cover_range[other] = {begin, Index(cover.size())};
      stack.push_back(Index(other));
    }
  }
  for (const auto seen : reached)
    if (!seen)
      return Index(-1);
  // an even winding moves the flood but covers nothing: each region keeps
  // its odd runs, in place, once no region reads its parent any more
  for (auto &range : cover_range) {
    auto kept = range[0];
    for (auto at = range[0]; at != range[1]; ++at)
      if ((cover_winding[std::size_t(at)] & 1) != 0) {
        cover[std::size_t(kept)] = cover[std::size_t(at)];
        cover_winding[std::size_t(kept)] = cover_winding[std::size_t(at)];
        ++kept;
      }
    range[1] = kept;
  }
  return n_regions;
}

} // namespace tf::arrangement
