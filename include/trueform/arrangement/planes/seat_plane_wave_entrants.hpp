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
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./classify_plane_wave_entrants.hpp"
#include "./plane_piece_key.hpp"
#include "./plane_tier_definitions.hpp"
#include "./port_plane_diff.hpp"
#include "./state_plane_group_carriers.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

namespace tf::arrangement {

/// Seat a round's promoted faces in this arrangement's own tier.
///
/// THE ENTRANT GETS A NEW SPAN in the local tables — every edge that is its
/// alone. The ONE row that cannot go there is the SHARED edge: its canonical
/// group already exists, the group IS the cross-face join, and a second group
/// for one wall is the twin-wall defect. That row JOINS the existing group as
/// one more instance — appended at the span's tail, sorted last, every other
/// position untouched, and no correspondence table. The rider is that one
/// row; @ref tf::arrangement::port_plane_diff is where it lands, because the
/// port is the operation that moves a group from the world tier to this one.
///
/// So the order is the whole algorithm: the shared sides name the groups the
/// port TAKES and ride them in; the sides that name nothing found local
/// groups of their own after it; and each promoted face's block is then its
/// rows in KEY order, which is the invariant every block consumer indexes by.
/// A promoted plane must hold a local ticket — the world tables end at the
/// world's own carriers — so the ticket space grows with the faces.
///
/// Every group the entrants name must be seatable: `route_of` states that,
/// and a caller states only faces @ref
/// tf::arrangement::classify_plane_wave_entrants passed. False rejects the
/// call; the extents it states are the ones this append itself indexes.
template <typename Index, typename Int, typename World, typename PlaneOfFace>
auto seat_plane_wave_entrants(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &raw_entrants,
    const tf::buffer<Index> &world_group_of, const tf::buffer<int> &route_of,
    Index entrant_plane_base, const PlaneOfFace &plane_of_face,
    tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    tf::buffer<Index> &plane_ticket, tf::buffer<Index> &group_router) -> bool {
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  const auto &world_tables = world.tables();
  const auto n_groups = raw_entrants.n_canon();
  const auto n_entrant_planes = Index(raw_entrants.edges().size());
  const auto index_extent = std::size_t(std::numeric_limits<Index>::max());
  if (n_entrant_planes == Index(0))
    return true;
  if (world_group_of.size() != std::size_t(n_groups) ||
      route_of.size() != std::size_t(n_groups) ||
      entrant_plane_base < Index(0) ||
      std::size_t(entrant_plane_base) >
          index_extent - std::size_t(n_entrant_planes))
    return false;
  // A local tier and an empty router is a state no consumer accepts, and the
  // fresh groups below can create one where the port would not.
  if (group_router.size() == 0) {
    group_router.allocate(std::size_t(world_tables.n_canon()));
    tf::parallel_fill(group_router, Index(-1));
  }

  // THE RIDERS, carried by value in the span's own order: provenance then
  // flags, which is what the port appends them under and what every later
  // search of a span assumes.
  struct rider_t {
    Index group;
    def_t def;
    Index row;
  };
  const auto &entrant_defs = raw_entrants.defs();
  const auto &entrant_def_offsets = raw_entrants.def_offsets();
  tf::buffer<rider_t> riders;
  tf::sequenced_generate(
      tf::make_sequence_range(n_groups), riders,
      [&](Index group, tf::buffer<rider_t> &out) {
        if (route_of[std::size_t(group)] != plane_wave_entrant_rides_port)
          return;
        const auto joined = world_group_of[std::size_t(group)];
        for (auto row = entrant_def_offsets[std::size_t(group)];
             row != entrant_def_offsets[std::size_t(group) + 1]; ++row)
          out.push_back({joined, entrant_defs[std::size_t(row)], row});
      },
      tf::checked);
  tbb::parallel_sort(
      riders.begin(), riders.end(), [](const rider_t &x, const rider_t &y) {
        if (x.group != y.group)
          return x.group < y.group;
        if (tf::intersect::graph::plane_def_instance_less(x.def, y.def))
          return true;
        if (tf::intersect::graph::plane_def_instance_less(y.def, x.def))
          return false;
        return x.row < y.row;
      });

  tf::buffer<Index> taken;
  tf::buffer<Index> rider_offsets;
  rider_offsets.push_back(Index(0));
  for (std::size_t at = 0; at != riders.size(); ++at)
    if (at == 0 || riders[at - 1].group != riders[at].group) {
      taken.push_back(riders[at].group);
      if (at != 0)
        rider_offsets.push_back(Index(at));
    }
  if (riders.size() != 0)
    rider_offsets.push_back(Index(riders.size()));

  // the port repoints every carrier of a taken group, so the diff is that
  // group's own carriers and nothing wider
  tf::buffer<Index> diff_planes;
  state_plane_group_carriers(world_tables, tf::make_range(taken), plane_of_face,
                             diff_planes);
  if (diff_planes.size() != 0) {
    tbb::parallel_sort(diff_planes.begin(), diff_planes.end());
    diff_planes.erase_till_end(
        std::unique(diff_planes.begin(), diff_planes.end()));
  }

  const auto group_base = local_tables.n_canon();
  if (!port_plane_diff(
          world_tables, tf::make_range(diff_planes), taken, local_tables,
          plane_ticket, group_router, rider_offsets,
          tf::make_mapped_range(tf::make_range(riders),
                                [](const rider_t &rider) -> const def_t & {
                                  return rider.def;
                                })))
    return false;

  // THE ROW EACH ENTRANT DEFINITION SURVIVES AS: a rider is the tail of the
  // span the port just minted, at its own rank inside the run.
  tf::buffer<Index> seat;
  seat.allocate_and_initialize(entrant_defs.size(), Index(-1));
  auto &local_defs = local_tables.defs();
  auto &local_def_offsets = local_tables.def_offsets();
  for (std::size_t rank = 0; rank != taken.size(); ++rank) {
    const auto mint = group_base + Index(rank);
    const auto run = std::size_t(rider_offsets[rank + 1] - rider_offsets[rank]);
    const auto tail =
        std::size_t(local_def_offsets[std::size_t(mint) + 1]) - run;
    for (std::size_t at = 0; at != run; ++at)
      seat[std::size_t(riders[std::size_t(rider_offsets[rank]) + at].row)] =
          Index(tail + at);
  }

  // a side whose key names nothing the world holds founds a group of its own
  if (local_def_offsets.size() == 0)
    local_def_offsets.push_back(Index(0));
  for (Index group = 0; group < n_groups; ++group) {
    if (route_of[std::size_t(group)] != plane_wave_entrant_fresh)
      continue;
    const auto begin = std::size_t(entrant_def_offsets[std::size_t(group)]);
    const auto end = std::size_t(entrant_def_offsets[std::size_t(group) + 1]);
    const auto mint = local_tables.n_canon();
    const auto at = local_defs.size();
    if (end - begin > index_extent - at ||
        std::size_t(mint) == index_extent)
      return false;
    local_defs.reallocate(at + (end - begin));
    for (auto row = begin; row != end; ++row) {
      auto minted = entrant_defs[row];
      minted.id = mint;
      local_defs[at + (row - begin)] = minted;
      seat[row] = Index(at + (row - begin));
    }
    local_def_offsets.push_back(Index(local_defs.size()));
    local_tables.n_canon() = mint + Index(1);
  }

  // THE PROMOTED PLANE MUST HOLD A LOCAL TICKET: the world tables end at the
  // world's own carriers, so a suffix plane reading `-1` names nothing
  const auto n_planes =
      std::size_t(entrant_plane_base) + std::size_t(n_entrant_planes);
  plane_ticket.reallocate_and_initialize(n_planes, Index(-1));

  auto &block_offsets = local_tables.edges().offsets_buffer();
  auto &block_rows = local_tables.edges().data_buffer();
  const auto entrant_rows = raw_entrants.edges().data_buffer().size();
  if (entrant_rows > index_extent - block_rows.size() ||
      std::size_t(n_entrant_planes) > index_extent - block_offsets.size())
    return false;
  if (block_offsets.size() == 0)
    block_offsets.push_back(Index(0));
  const auto block_base = Index(local_tables.edges().size());
  const auto row_base = block_rows.size();
  const auto old_block_offsets = block_offsets.size();
  block_offsets.reallocate(old_block_offsets + std::size_t(n_entrant_planes));
  block_rows.reallocate(row_base + entrant_rows);
  const auto tier =
      make_plane_tier_definitions(world_tables, local_tables, true);
  tf::buffer<std::array<Index, 5>> keyed;
  auto write = row_base;
  for (Index face = 0; face < n_entrant_planes; ++face) {
    const auto begin = write;
    for (const auto row : raw_entrants.plane_edges(face)) {
      if (seat[std::size_t(row)] == Index(-1))
        return false;
      block_rows[write++] = seat[std::size_t(row)];
    }
    sort_plane_block_by_key(tier, block_rows, begin, write, keyed);
    block_offsets[old_block_offsets + std::size_t(face)] = Index(write);
    plane_ticket[std::size_t(entrant_plane_base + face)] = block_base + face;
  }
  return true;
}

} // namespace tf::arrangement
