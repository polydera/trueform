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

/// Which of a pooled plane's members cover which of its regions.
///
/// The triangulation states the regions — its components, region 0 the
/// hull exterior — so the only fact left is a parity: a member's own
/// side is a closed cycle of walls, and every wall belongs to exactly
/// the one member whose instance stated it, so crossing it toggles that
/// member alone. One flood from the exterior, where nobody covers
/// anything, settles all of them.
///
/// A region's cover is therefore its parent's cover TOGGLED by the wall
/// between them, and both are ascending runs of members, so the step is one
/// merge and the whole table costs what its facts cost. `cover_range` is the
/// dense side: per region a ticket into `cover`, the exterior's empty.
/// Returns n_regions, or -1 for a carrier whose regions do not all reach the
/// exterior — a parity is only defined along a path from it.
template <typename Index, typename Cdt>
auto flood_plane_member_coverage(
    const Cdt &cdt, const tf::buffer<std::array<Index, 2>> &cons_row,
    const tf::buffer<plane_member_statement<Index>> &cons_statements,
    Index n_constraints, tf::buffer<std::array<Index, 3>> &adjacency,
    tf::buffer<Index> &offsets, tf::buffer<Index> &stack,
    tf::buffer<char> &reached, tf::buffer<std::array<Index, 2>> &cover_range,
    tf::buffer<Index> &cover) -> Index {
  const auto labels = cdt.region_labels();
  Index n_regions = 1;
  for (const auto label : labels)
    if (label >= n_regions)
      n_regions = label + Index(1);

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
      const auto wall = owners[std::size_t(e)].input_id == Index(-1)
                            ? n_constraints
                            : owners[std::size_t(e)].input_id;
      adjacency.push_back({label, other, wall});
      if (neighbor == Index(-1))
        adjacency.push_back({other, label, wall});
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
      const auto other = std::size_t(adjacency[std::size_t(i)][1]);
      if (reached[other])
        continue;
      reached[other] = 1;
      const auto parent = cover_range[std::size_t(region)];
      const auto wall = cons_row[std::size_t(adjacency[std::size_t(i)][2])];
      const auto begin = Index(cover.size());
      auto held = parent[0];
      auto toggled = wall[0];
      for (;;) {
        while (toggled != wall[1] &&
               cons_statements[std::size_t(toggled)].parity == char(0))
          ++toggled;
        if (held == parent[1] && toggled == wall[1])
          break;
        if (toggled == wall[1] ||
            (held != parent[1] &&
             cover[std::size_t(held)] <
                 cons_statements[std::size_t(toggled)].member)) {
          const auto member = cover[std::size_t(held)];
          cover.push_back(member);
          ++held;
          continue;
        }
        const auto member = cons_statements[std::size_t(toggled)].member;
        if (held != parent[1] && cover[std::size_t(held)] == member) {
          ++held;
          ++toggled;
          continue;
        }
        cover.push_back(member);
        ++toggled;
      }
      cover_range[other] = {begin, Index(cover.size())};
      stack.push_back(Index(other));
    }
  }
  for (const auto seen : reached)
    if (!seen)
      return Index(-1);
  return n_regions;
}

} // namespace tf::arrangement
