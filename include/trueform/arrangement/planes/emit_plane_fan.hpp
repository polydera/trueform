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

#include "../../core/range.hpp"
#include "../../topology/topo_id.hpp"
#include "../../topology/triangulation/for_each_convex_chain_fan_triangle.hpp"
#include "./find_plane_carrier_fan.hpp"
#include "./state_plane_member_point_subs.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace tf::arrangement {

/// CORE. Emit one carrier of the convex family, over the ring its predicate
/// found. Emitting the ring IS the source winding, so the only orientation
/// question is the one the predicate already answered.
///
/// Every fact the ordinary emission states, this states from the prepared
/// records directly, because the ring is the point table: a corner's sub is
/// its own row of it, and a slot's piece is the constraint whose endpoints it
/// spans — a lookup over at most four, since each is a distinct pair. The
/// whole fan lies in ONE cell: no constraint separates its triangles, and a
/// cell id is plane-local and renumbered by first appearance, so `0` names it
/// completely.
template <typename Index, typename Int, typename Local>
auto emit_plane_fan(Local &local, const plane_carrier_fan<Index> &fan,
                    bool record_arrangement, bool record_cells) -> void {
  const auto before = local.tris.size();
  state_plane_point_subs(local.cons, local.cons_side, local.ends.size(),
                         local.point_subs, local.subs_touched);
  const auto piece_of = [&local](Index a, Index b) {
    for (std::size_t at = 0; at < local.cons.size() / 2; ++at) {
      const auto x = local.cons[2 * at];
      const auto y = local.cons[2 * at + 1];
      if ((x == a && y == b) || (x == b && y == a))
        return local.cons_group[at];
    }
    return Index(-1);
  };
  const auto ring = tf::make_range(
      fan.ring.begin(), fan.ring.begin() + std::ptrdiff_t(fan.size));
  tf::topology::for_each_convex_chain_fan_triangle(
      ring, fan.apex, fan.size, [&](Index apex, Index a, Index b) {
        std::array<Index, 3> points{apex, a, b};
        if (fan.reversed)
          std::swap(points[1], points[2]);
        std::array<Index, 3> corners;
        std::array<tf::topo_id<short>, 3> subs;
        for (int corner = 0; corner < 3; ++corner) {
          corners[std::size_t(corner)] =
              local.ends[std::size_t(points[std::size_t(corner)])];
          subs[std::size_t(corner)] =
              local.point_subs[std::size_t(points[std::size_t(corner)])];
        }
        if (record_arrangement) {
          std::array<Index, 3> slots;
          for (std::size_t slot = 0; slot < 3; ++slot)
            slots[slot] = piece_of(points[slot], points[(slot + 1) % 3]);
          local.slots.push_back(slots);
          local.coplanar_of.push_back(Index(-1));
          local.stacked.push_back(char(0));
        }
        local.subs.push_back(subs);
        local.tris.push_back(corners);
        if (record_cells)
          local.cell_of.push_back(Index(0));
      });
  local.face_range.push_back({Index(before), Index(local.tris.size())});
}

} // namespace tf::arrangement
