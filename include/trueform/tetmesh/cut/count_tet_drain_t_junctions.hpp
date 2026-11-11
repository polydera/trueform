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
#include "../../core/config/verify.hpp"
#include "./tet_changed_triangle.hpp"
#include "./tet_slot_holders.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// The T-junctions a drain's split leaves: every point it put on an edge
/// (`edge_points`, read through `speaks`, where the wave's welds leave it)
/// must be a corner of every live triangle holding that edge that does not
/// hold it as a corner already — a changed triangle whose pieces hold it,
/// or whose line portion does; a changed triangle every piece of which
/// dropped holds no edge. THE LINE LAW: a point two sides of its holder
/// state is one identity on both, and both sides run from their shared
/// corner through it, so between them the two sides are one edge — a line,
/// which holds the point and emits nothing; the holders across both sides
/// hold that stretch once each way. A holder that lacks one is the drain's
/// to refuse. Under @ref tf::core::verification no piece holds a point
/// `speaks` retires. `edge_key(t, k)` is side `k` of triangle `t` in the
/// slots `edge_points` is sorted by. O(pieces + Σ holders of the edges
/// points stand on × their pieces + the rows on an unheld holder's sides).
template <typename Index, typename Live, typename Holds, typename CornersOf,
          typename EdgeKey, typename Speaks>
auto count_tet_drain_t_junctions(
    const tf::buffer<std::array<Index, 3>> &edge_points,
    const tet_slot_holders<Index> &holders, const Live &live,
    const Holds &holds, const CornersOf &corners_of, const EdgeKey &edge_key,
    const Speaks &speaks,
    const tf::buffer<tet_changed_triangle<Index>> &changed,
    const tf::buffer<std::array<Index, 3>> &pieces) -> std::size_t {
  if constexpr (tf::core::verification)
    for (const auto &piece : pieces)
      for (auto v : piece)
        tf::core::verify(speaks(v) == v, "no piece holds a retired point");
  const auto states = [&](Index u, std::size_t k, Index p) {
    const auto key = edge_key(u, k);
    for (auto at =
             std::lower_bound(edge_points.begin(), edge_points.end(),
                              std::array<Index, 3>{key[0], key[1], Index(-1)});
         at != edge_points.end() && (*at)[0] == key[0] && (*at)[1] == key[1];
         ++at)
      if (speaks((*at)[2]) == p)
        return true;
    return false;
  };
  std::size_t n = 0;
  for (const auto &entry : edge_points) {
    const Index p = speaks(entry[2]);
    for_each_tet_slot_holder(holders, entry[0], [&](Index u) {
      if (!live(std::size_t(u)) || !holds(u, entry[1]))
        return;
      auto corners = corners_of(u);
      for (auto &corner : corners)
        corner = speaks(corner);
      if (p == corners[0] || p == corners[1] || p == corners[2])
        return;
      const auto found =
          std::lower_bound(changed.begin(), changed.end(), u,
                           [](const tet_changed_triangle<Index> &c, Index key) {
                             return c.triangle < key;
                           });
      const bool member = found != changed.end() && found->triangle == u;
      if (member && found->first == found->last)
        return;
      bool held = false;
      if (member)
        for (Index i = found->first; i < found->last && !held; ++i)
          for (auto v : pieces[std::size_t(i)])
            held = held || v == p;
      if (!held)
        held =
            std::size_t(states(u, 0, p)) + states(u, 1, p) + states(u, 2, p) >=
            2;
      n += !held;
    });
  }
  return n;
}

} // namespace tf::tetmesh
