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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_group_router.hpp"

#include <cstddef>

namespace tf::arrangement {

/// A promoted face's side that JOINS a standing group instead of founding
/// one, and the route by which it joins.
///
/// THE ENTRANT GETS A NEW SPAN in the local tables — every edge that is its
/// alone. The ONE row that cannot go there is the SHARED edge: its canonical
/// group already exists — the neighbour's instance and the wave's split live
/// on it, the group IS the cross-face join, and a second group for one wall
/// is the twin-wall defect. That row therefore joins the existing group as
/// one more instance. The rider is that one row, and everything below is the
/// four routes one row can take, not four mechanisms.
/// The key names no group the world holds: the row founds a local group of
/// its own, with nothing to join.
inline constexpr int plane_wave_entrant_fresh = 0;
/// The key names a world group no wave has changed: the row rides the port
/// that takes it.
inline constexpr int plane_wave_entrant_rides_port = 1;
/// The key names a group a wave already took, or a root a split already
/// retired. The join would have to grow a span the local tier has already
/// closed, which is a second merge — the face is declined whole.
inline constexpr int plane_wave_entrant_declined = 2;

/// Route every raw entrant group: `world_group_of` is the world canonical
/// group it joins, `-1` when it founds one, and `route_of` is the verdict
/// above.
template <typename Index, typename Int, typename World>
auto classify_plane_wave_entrants(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &raw_entrants,
    const tf::buffer<Index> &group_router, tf::buffer<Index> &world_group_of,
    tf::buffer<int> &route_of) -> void {
  const auto &world_tables = world.tables();
  const auto n_groups = raw_entrants.n_canon();
  world_group_of.allocate(std::size_t(n_groups));
  route_of.allocate(std::size_t(n_groups));
  tf::parallel_for_each(
      tf::make_sequence_range(n_groups),
      [&](Index group) {
        const auto found = tf::intersect::graph::find_plane_canon_group(
            world_tables, raw_entrants.canon_group(group)[0]);
        world_group_of[std::size_t(group)] = found;
        if (found == Index(-1)) {
          route_of[std::size_t(group)] = plane_wave_entrant_fresh;
          return;
        }
        route_of[std::size_t(group)] =
            resolve_plane_group_router(group_router, found, Index(-1)) ==
                    Index(-1)
                ? plane_wave_entrant_rides_port
                : plane_wave_entrant_declined;
      },
      tf::checked);
}

/// Which entrant FACES can be seated: a face enters WHOLE — its corner cycle
/// is its loop — so it is seatable exactly when every row of its block is.
/// One char per entrant face, in the entrant table's own block order.
template <typename Index, typename Int>
auto state_plane_wave_entrant_seats(
    const tf::intersect::graph::plane_tables<Index, Int> &raw_entrants,
    const tf::buffer<int> &route_of, tf::buffer<char> &face_seatable) -> void {
  face_seatable.allocate(raw_entrants.edges().size());
  tf::parallel_fill(face_seatable, char(1));
  tf::parallel_for_each(
      tf::make_sequence_range(Index(raw_entrants.edges().size())),
      [&](Index face) {
        for (const auto row : raw_entrants.plane_edges(face))
          if (route_of[std::size_t(
                  raw_entrants.edge_defs()[std::size_t(row)].id)] ==
              plane_wave_entrant_declined) {
            face_seatable[std::size_t(face)] = char(0);
            return;
          }
      },
      tf::checked);
}

} // namespace tf::arrangement
