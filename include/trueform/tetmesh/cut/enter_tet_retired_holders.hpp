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
#include "../../intersect/graph/plane_uncut_entrants.hpp"
#include "./tet_changed_triangle.hpp"
#include "./tet_slot_holders.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// THE ENTRANCE of a drain, the law the arrangement's tiers obey
/// (@ref tf::intersect::graph::discover_weld_entrants): a point the wave
/// retires (`speaks(p) != p`) is a corner of one of the wave's `members`,
/// and every live triangle holding its slot that the wave did not change
/// enters the drain, restated with every corner speaking its survivor —
/// one piece, or none where two of its corners became one identity.
/// The entrants are the arrangement's own discovery over the slot holders,
/// a face entering where the drain's world does not state it already: dead,
/// or changed. `changed`, ascending, gains the entrants and stays
/// ascending; `retired` and `entered` are scratch. O(the members' corners
/// log + the holders of the retired slots).
template <typename Index, typename Faces, typename CornersOf, typename Live,
          typename Speaks>
auto enter_tet_retired_holders(const tf::buffer<Index> &members,
                               const Faces &faces, const CornersOf &corners_of,
                               const tet_slot_holders<Index> &holders,
                               const Live &live, const Speaks &speaks,
                               tf::buffer<tet_changed_triangle<Index>> &changed,
                               tf::buffer<std::array<Index, 3>> &pieces,
                               tf::buffer<Index> &retired,
                               tf::buffer<Index> &entered) -> void {
  const auto by_triangle = [](const tet_changed_triangle<Index> &a,
                              const tet_changed_triangle<Index> &b) {
    return a.triangle < b.triangle;
  };
  const auto is_changed = [&changed](Index u) {
    const auto at = std::lower_bound(changed.begin(), changed.end(), u,
                                     [](const tet_changed_triangle<Index> &x,
                                        Index y) { return x.triangle < y; });
    return at != changed.end() && at->triangle == u;
  };
  // the retired slots, once each, from the members' corners
  retired.clear();
  for (const auto t : members) {
    const auto corners = corners_of(t);
    for (std::size_t k = 0; k < 3; ++k)
      if (speaks(corners[k]) != corners[k])
        retired.push_back(Index(faces[std::size_t(t)][k]));
  }
  std::sort(retired.begin(), retired.end());
  retired.erase_till_end(std::unique(retired.begin(), retired.end()));
  entered.clear();
  tf::intersect::graph::discover_weld_entrants<Index>(
      retired,
      [&holders](Index slot, const auto &f) {
        for_each_tet_slot_holder(holders, slot, [&f](Index u) { f(u, u); });
      },
      [&](Index u) { return !live(std::size_t(u)) || is_changed(u); }, entered);
  std::sort(entered.begin(), entered.end());
  entered.erase_till_end(std::unique(entered.begin(), entered.end()));
  for (auto u : entered) {
    const auto corners = corners_of(u);
    const std::array<Index, 3> piece{speaks(corners[0]), speaks(corners[1]),
                                     speaks(corners[2])};
    const Index first = Index(pieces.size());
    if (piece[0] != piece[1] && piece[1] != piece[2] && piece[0] != piece[2])
      pieces.push_back(piece);
    changed.push_back({u, first, Index(pieces.size())});
  }
  std::sort(changed.begin(), changed.end(), by_triangle);
}

} // namespace tf::tetmesh
