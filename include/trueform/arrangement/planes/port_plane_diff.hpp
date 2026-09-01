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

#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/none.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_definition_source.hpp"
#include "./plane_group_router.hpp"
#include "./plane_tier_definitions.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace tf::arrangement {

/// Take the wave's CHANGED groups into this arrangement's tier, and give every
/// plane of the diff a local block.
///
/// THE OWNERSHIP LAW: a group whose statement CHANGES goes local, and it takes
/// every carrier of it in the same wave — so a group this arrangement owns has
/// no carrier still reading the world, and one identity is published the same
/// way on every plane that states it.
///
/// AN UNCHANGED GROUP STAYS THE WORLD'S, VERBATIM, FOR BOTH CARRIERS. A block
/// this call writes keeps naming the world's own rows for every group the wave
/// did not change, so the carrier it ports and the carrier it never touched
/// read one definition, and the ticket both publish is the immutable root's.
/// That is what makes the frontier a RING — the changed groups' carriers — and
/// never the component.
///
/// A taken group goes WHOLE: its canonical span moves instance for instance,
/// every carrier included, so the local span is complete the moment it exists.
///
/// AND IT MAY GAIN RIDERS. A face the cut world never named still holds the
/// edge, and the entrance that promotes it has exactly one row it cannot give
/// a span of its own: the SHARED side, whose canonical group already exists —
/// the neighbour's instance and the wave's split live on it, and the group IS
/// the cross-face join, so a second group for one wall is the twin-wall
/// defect. That row therefore JOINS this group as one more instance. The
/// sentence is the local arrangement's own, said here at the tier that ports:
/// @ref tf::intersect::graph::respan_plane_defs — *"extra instances of a cut
/// group ride its piece emission"* — and the build, the refinement and the
/// wave are the ONE entrance's three occasions of it.
///
/// THE MAP IS THE RANK, AND THE RIDERS DO NOT MOVE IT. A span moves in order,
/// so a world row becomes
///
///     local_row = local_def_offsets[router[group]] +
///                 (world_row - world_def_offsets[group])
///
/// which is total by OWNERSHIP. A rider's face is a stamp PAST every face the
/// world holds, so it sorts last under the span's own order — provenance then
/// flags — and appending the riders at the span's TAIL leaves every world
/// row's rank exactly where it was. No correspondence table exists and none
/// may be introduced.
///
/// The riders are stated per taken group by `rider_offsets`, aligned with
/// `taken_groups`, each run already in span order. A caller that promotes
/// nothing states neither, and the channel is not compiled.
///
/// False rejects the call and publishes nothing: the two tiers' interiors were
/// proven by the producers that appended them, so this states only what the
/// append and the rank formula themselves index — including the diff's own
/// shape, ascending and unique inside the plane extent, the taken set's shape
/// inside the immutable canonical extent, and the tickets it reads "already
/// local" from, which must name a block the local tier holds.
template <typename Index, typename Int, typename DiffPlanes,
          typename RiderOffsets = tf::none_t, typename Riders = tf::none_t>
auto port_plane_diff(
    const tf::intersect::graph::plane_tables<Index, Int> &world_tables,
    const DiffPlanes &diff_planes, const tf::buffer<Index> &taken_groups,
    tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    tf::buffer<Index> &plane_ticket, tf::buffer<Index> &group_router,
    const RiderOffsets &rider_offsets = tf::none,
    const Riders &riders = tf::none) -> bool {
  const auto index_extent = std::size_t(std::numeric_limits<Index>::max());
  const auto n_world_planes = Index(world_tables.edges().size());
  const auto n_immutable = world_tables.n_canon();
  if (plane_ticket.size() > index_extent ||
      world_tables.defs().size() > index_extent)
    return false;
  const auto n_planes = Index(plane_ticket_space(world_tables, plane_ticket));
  if (group_router.size() != 0 &&
      group_router.size() != std::size_t(n_immutable))
    return false;
  if (tf::parallel_contains(
          tf::make_sequence_range(diff_planes.size()),
          [&](std::size_t at) {
            const auto plane = diff_planes[at];
            return plane < Index(0) || plane >= n_planes ||
                   (at != 0 && diff_planes[at - 1] >= plane) ||
                   (plane >= n_world_planes &&
                    (plane_ticket.size() == 0 ||
                     plane_ticket[std::size_t(plane)] == Index(-1)));
          },
          tf::checked))
    return false;
  if (!local_tables.well_formed())
    return false;
  const auto n_local_blocks = Index(local_tables.edges().size());
  if (!plane_ticket_well_formed(plane_ticket, n_world_planes, n_local_blocks))
    return false;

  const auto routed = [&group_router](Index group) {
    return resolve_plane_group_router(group_router, group, Index(-1));
  };
  if (tf::parallel_contains(
          tf::make_sequence_range(taken_groups.size()),
          [&](std::size_t at) {
            const auto group = taken_groups[at];
            return group < Index(0) || group >= n_immutable ||
                   (at != 0 && taken_groups[at - 1] >= group) ||
                   routed(group) != Index(-1);
          },
          tf::checked))
    return false;
  if constexpr (!std::is_same<RiderOffsets, tf::none_t>::value)
    if (rider_offsets.size() != taken_groups.size() + 1 ||
        rider_offsets[0] != Index(0) ||
        rider_offsets[taken_groups.size()] != Index(riders.size()) ||
        tf::parallel_contains(
            tf::make_sequence_range(taken_groups.size()),
            [&](std::size_t rank) {
              return rider_offsets[rank] > rider_offsets[rank + 1];
            },
            tf::checked))
      return false;

  tf::buffer<Index> ported_planes;
  tf::sequenced_generate(
      diff_planes, ported_planes,
      [&](Index plane, tf::buffer<Index> &out) {
        if (plane < n_world_planes &&
            (plane_ticket.size() == 0 ||
             plane_ticket[std::size_t(plane)] == Index(-1)))
          out.push_back(plane);
      },
      tf::checked);
  if (ported_planes.size() == 0 && taken_groups.size() == 0)
    return true;

  const auto world_defs = world_tables.edge_defs();
  const auto &world_def_offsets = world_tables.def_offsets();
  tf::buffer<Index> block_prefix;
  block_prefix.allocate(ported_planes.size() + 1);
  block_prefix[0] = Index(0);
  for (std::size_t at = 0; at < ported_planes.size(); ++at) {
    const auto count = world_tables.plane_edges(ported_planes[at]).size();
    if (count > index_extent - std::size_t(block_prefix[at]))
      return false;
    block_prefix[at + 1] = block_prefix[at] + Index(count);
  }
  tf::buffer<Index> named_groups;
  named_groups.allocate(std::size_t(block_prefix[ported_planes.size()]));
  tf::parallel_for_each(
      tf::make_sequence_range(ported_planes.size()), [&](std::size_t at) {
        auto row = std::size_t(block_prefix[at]);
        for (const auto world_row : world_tables.plane_edges(ported_planes[at]))
          named_groups[row++] = world_defs[std::size_t(world_row)].id;
      });
  if (named_groups.size() != 0) {
    tbb::parallel_sort(named_groups.begin(), named_groups.end());
    named_groups.erase_till_end(
        std::unique(named_groups.begin(), named_groups.end()));
  }
  // THE OWNERSHIP LAW: a plane this wave ports for the first time names no
  // group an earlier wave took — taking a group took every carrier of it
  if (tf::parallel_contains(
          tf::make_range(named_groups),
          [&](Index group) { return routed(group) != Index(-1); },
          tf::checked))
    return false;

  // EVERY fallible extent is proven before ANY publication — false must
  // leave the tier, the ticket, and the router exactly as they arrived
  auto &local_defs = local_tables.defs();
  auto &local_def_offsets = local_tables.def_offsets();
  const auto group_base = local_tables.n_canon();
  const auto def_row_base = Index(local_defs.size());
  if (taken_groups.size() > index_extent - std::size_t(group_base))
    return false;
  tf::buffer<Index> span_offsets;
  span_offsets.allocate(taken_groups.size() + 1);
  span_offsets[0] = Index(0);
  for (std::size_t rank = 0; rank < taken_groups.size(); ++rank) {
    const auto group = std::size_t(taken_groups[rank]);
    auto count = std::size_t(world_def_offsets[group + 1] -
                             world_def_offsets[group]);
    if constexpr (!std::is_same<RiderOffsets, tf::none_t>::value)
      count += std::size_t(rider_offsets[rank + 1] - rider_offsets[rank]);
    if (count > index_extent - std::size_t(def_row_base) -
                    std::size_t(span_offsets[rank]))
      return false;
    span_offsets[rank + 1] = span_offsets[rank] + Index(count);
  }
  if (ported_planes.size() >
          index_extent - std::size_t(local_tables.edges().size()) ||
      std::size_t(block_prefix[ported_planes.size()]) >
          index_extent -
              std::size_t(local_tables.edges().data_buffer().size()))
    return false;

  if (plane_ticket.size() == 0) {
    plane_ticket.allocate(std::size_t(n_planes));
    tf::parallel_fill(plane_ticket, Index(-1));
  }
  if (group_router.size() == 0) {
    group_router.allocate(std::size_t(n_immutable));
    tf::parallel_fill(group_router, Index(-1));
  }

  if (taken_groups.size() != 0) {
    local_defs.reallocate(std::size_t(def_row_base) +
                          std::size_t(span_offsets[taken_groups.size()]));
    if (local_def_offsets.size() == 0)
      local_def_offsets.push_back(Index(0));
    const auto old_def_offsets = local_def_offsets.size();
    local_def_offsets.reallocate(old_def_offsets + taken_groups.size());
    tf::parallel_for_each(
        tf::make_sequence_range(taken_groups.size()),
        [&](std::size_t rank) {
          const auto group = taken_groups[rank];
          const auto mint = group_base + Index(rank);
          auto at = std::size_t(def_row_base + span_offsets[rank]);
          for (const auto &def : world_tables.canon_group(group)) {
            auto minted = def;
            minted.id = mint;
            local_defs[at++] = minted;
          }
          // the riders close the span: their faces stamp past every world
          // face, so the span stays ordered by provenance and every world
          // row keeps the rank the map states
          if constexpr (!std::is_same<RiderOffsets, tf::none_t>::value)
            for (auto rider = std::size_t(rider_offsets[rank]);
                 rider != std::size_t(rider_offsets[rank + 1]); ++rider) {
              auto minted = riders[rider];
              minted.id = mint;
              assert(at == std::size_t(def_row_base + span_offsets[rank]) ||
                     !tf::intersect::graph::plane_def_instance_less(
                         minted, local_defs[at - 1]));
              local_defs[at++] = minted;
            }
          local_def_offsets[old_def_offsets + rank] =
              def_row_base + span_offsets[rank + 1];
          group_router[std::size_t(group)] = mint;
        });
    local_tables.n_canon() = group_base + Index(taken_groups.size());
  }

  // the one row a block of this arrangement names a world row by: the rank
  // inside the span the router named it, or the world's own row CARRIED
  const auto current_row = [&](Index world_row) {
    const auto &def = world_defs[std::size_t(world_row)];
    const auto mint = group_router[std::size_t(def.id)];
    if (mint == Index(-1))
      return plane_tier_definitions<Index, Int>::carried(world_row);
    const auto local_row =
        local_def_offsets[std::size_t(mint)] +
        (world_row - world_def_offsets[std::size_t(def.id)]);
    assert(local_defs[std::size_t(local_row)].point_tag_0 == def.point_tag_0 &&
           local_defs[std::size_t(local_row)].point_0 == def.point_0 &&
           local_defs[std::size_t(local_row)].point_tag_1 == def.point_tag_1 &&
           local_defs[std::size_t(local_row)].point_1 == def.point_1);
    return local_row;
  };

  auto &block_offsets = local_tables.edges().offsets_buffer();
  auto &block_rows = local_tables.edges().data_buffer();
  // A plane the tier already holds keeps its block and its order: a taken
  // group moved verbatim, so only the rows naming it are restated.
  if (taken_groups.size() != 0)
    tf::parallel_for_each(
        diff_planes,
        [&](Index plane) {
          const auto block = plane_ticket[std::size_t(plane)];
          if (block == Index(-1))
            return;
          for (auto at = std::size_t(block_offsets[std::size_t(block)]);
               at < std::size_t(block_offsets[std::size_t(block) + 1]); ++at)
            if (block_rows[at] < Index(0))
              block_rows[at] = current_row(
                  plane_tier_definitions<Index, Int>::carried(block_rows[at]));
        },
        tf::checked);

  if (ported_planes.size() == 0)
    return true;
  const auto block_base = Index(local_tables.edges().size());
  const auto data_base = Index(block_rows.size());
  if (block_offsets.size() == 0)
    block_offsets.push_back(Index(0));
  const auto old_block_offsets = block_offsets.size();
  block_offsets.reallocate(old_block_offsets + ported_planes.size());
  block_rows.reallocate(std::size_t(data_base) +
                        std::size_t(block_prefix[ported_planes.size()]));
  tf::parallel_for_each(
      tf::make_sequence_range(ported_planes.size()),
      [&](std::size_t at) {
        const auto plane = ported_planes[at];
        auto row = std::size_t(data_base + block_prefix[at]);
        for (const auto world_row : world_tables.plane_edges(plane))
          block_rows[row++] = current_row(world_row);
        block_offsets[old_block_offsets + at] = data_base + block_prefix[at + 1];
        plane_ticket[std::size_t(plane)] = block_base + Index(at);
      });
  return true;
}

} // namespace tf::arrangement
