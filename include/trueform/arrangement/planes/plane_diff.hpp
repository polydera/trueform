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

#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/flat_of_vertex.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_definition_source.hpp"
#include "./plane_flat_carriers.hpp"
#include "./plane_group_router.hpp"
#include "./plane_tier_definitions.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace tf::arrangement {

/// Gather the diff a wave consumes: the weld-touched rows of every plane, and
/// the plane frontier the wave works on.
///
/// THE SWEEP is one flat parallel pass at PLANE grain, and liveness is the
/// route: a plane walks the block its ticket names — the world tier while the
/// ticket is empty or `-1`, otherwise its local block — so a ticketed plane's
/// superseded world block is never read. The plane index is the join key, so
/// the rows come back as one offset block per plane with no sort at all. An
/// empty delta retires nothing, so it publishes no rows and allocates nothing;
/// `weld_row_offsets.size() == 0` is that answer, not a missing one.
///
/// THE SWEEP READS THE DIRTY RING, NOT THE WORLD, wherever the world can
/// name it: a carrier holding no row that names a retired identity has
/// nothing to publish, and @ref tf::arrangement::states_flat_carriers is
/// where a world says it can name those carriers by MEMBERSHIP. The ring is
/// then the carriers the world names for this wave's retired identities plus
/// the blocks the wave holds — the only rows that can name an identity the
/// world's own tier never states. Both arms visit the whole plane space, so
/// the offset block stays one per plane either way; only the block reads
/// differ, and a skipped plane is one the world proved empty.
///
/// `retired` is the wave's CHANGED set alone — the flat identities its merge
/// rows retire, ascending — and one binary search per endpoint is the whole
/// test. An endpoint flattens through
/// @ref tf::intersect::graph::flat_of_vertex, and the caller flattens its
/// merge rows through that same producer.
///
/// THE WELD ROWS ARE STATED IN THE PLANE'S PRE-PORT TIER: the sweep runs
/// before the wave ports, so a plane the wave then ports for the first time
/// has its rows named in WORLD coordinates. The consumer translates such a
/// row after the port by the rank formula —
/// `local_row = local_def_offsets[router[id]] + (row - world_def_offsets[id])`
/// — total by OWNERSHIP, because a ported plane names only groups that port
/// copied verbatim; rows of planes already local at sweep time need no
/// translation. No correspondence table exists and none may be introduced.
///
/// `planes` unions the weld-touched planes and the carriers of every split
/// edge `order_plane_recovery_splits` published. An edge is an edge — its
/// name is (tier, id) — and its current span is one lookup: a local id
/// answers itself, an original one answers through `group_router` (`-1`, or
/// an empty router, leaves it in the world tier). A group the router retired
/// has no live span, and rejects the call. Carriers resolve through
/// `plane_of_face`, whatever face they are — cut, uncut, any wave.
///
/// False rejects the call and publishes nothing: the tiers' interiors were
/// proven by the wave that appended them, so this states only what the sweep
/// itself indexes.
template <typename Index, typename Int, typename World, typename VertexOffsets,
          typename PlaneOfFace>
auto gather_plane_diff(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket,
    const tf::buffer<Index> &group_router, const tf::buffer<Index> &retired,
    const VertexOffsets &vertex_offsets, const tf::buffer<Index> &split_edge,
    const tf::buffer<Index> &split_tier, const PlaneOfFace &plane_of_face,
    tf::buffer<Index> &weld_row_offsets, tf::buffer<Index> &weld_row_data,
    tf::buffer<Index> &planes, tf::buffer<Index> &taken) -> bool {
  const auto &world_tables = world.tables();
  if (vertex_offsets.size() == 0)
    return false;
  const auto n_world_planes = Index(world_tables.edges().size());
  if (plane_ticket.size() >
      std::size_t(std::numeric_limits<Index>::max()))
    return false;
  const auto n_planes = Index(plane_ticket_space(world_tables, plane_ticket));
  const auto n_local_blocks = Index(local_tables.edges().size());
  if (!plane_ticket_well_formed(plane_ticket, n_world_planes, n_local_blocks))
    return false;
  if (retired.size() > 1 &&
      tf::parallel_contains(
          tf::make_sequence_range(std::size_t(1), retired.size()),
          [&](std::size_t at) { return retired[at - 1] >= retired[at]; },
          tf::checked))
    return false;
  const auto n_immutable = world_tables.n_canon();
  if (group_router.size() != 0 &&
      group_router.size() != std::size_t(n_immutable))
    return false;
  const auto routed_group = [&group_router](Index group) {
    return resolve_plane_group_router(group_router, group, Index(-1));
  };
  if (split_tier.size() != split_edge.size())
    return false;
  if (tf::parallel_contains(
          tf::make_sequence_range(split_edge.size()),
          [&](std::size_t at) {
            const auto group = split_edge[at];
            if (split_tier[at] == Index(1))
              return group < Index(0) || group >= local_tables.n_canon();
            if (split_tier[at] != Index(0))
              return true;
            if (group < Index(0) || group >= n_immutable)
              return true;
            const auto local = routed_group(group);
            return local < Index(-1) || local >= local_tables.n_canon();
          },
          tf::checked))
    return false;

  weld_row_offsets.clear();
  weld_row_data.clear();
  planes.clear();
  taken.clear();
  const auto n_immutable_canon = world_tables.n_canon();

  const auto is_retired = [&retired](Index flat) {
    const auto at = std::lower_bound(retired.begin(), retired.end(), flat);
    return at != retired.end() && *at == flat;
  };
  if (retired.size() != 0) {
    // THE RING, dense in the plane space it indexes: workers state the same
    // value into it, which is the whole write. A world that names no
    // carriers builds none and every plane reads its block, as before.
    tf::buffer<char> dirty;
    if constexpr (states_flat_carriers<World, Index>) {
      dirty.allocate_and_initialize(std::size_t(n_planes), char(0));
      tf::parallel_for_each(
          tf::make_range(retired),
          [&](Index flat) {
            for (const auto carrier : world.carriers_of_flat(flat))
              dirty[std::size_t(carrier)] = char(1);
          },
          tf::checked);
    }
    tf::generate_offset_blocks(
        tf::make_sequence_range(n_planes), weld_row_offsets, weld_row_data,
        [&](Index plane, tf::buffer<Index> &groups) {
          if constexpr (states_flat_carriers<World, Index>)
            if (dirty[std::size_t(plane)] == char(0) &&
                plane_reads_world_tier(plane_ticket, plane))
              return;
          const auto source = find_plane_definition_source(
              world_tables, local_tables, plane_ticket, plane);
          const auto tier = make_plane_tier_definitions(
              world_tables, local_tables, !source.immutable);
          for (const auto row : source.tables.plane_edges(source.block)) {
            const auto &def = tier[std::size_t(row)];
            if (is_retired(tf::intersect::graph::flat_of_vertex(
                    vertex_offsets, def.point_tag_0, def.point_0)) ||
                is_retired(tf::intersect::graph::flat_of_vertex(
                    vertex_offsets, def.point_tag_1, def.point_1)))
              groups.push_back(tier.ticket(row));
          }
        });
  }

  if (weld_row_offsets.size() != 0)
    tf::generic_generate(
        tf::make_sequence_range(n_planes), planes,
        [&](Index plane, tf::buffer<Index> &out) {
          if (weld_row_offsets[std::size_t(plane) + 1] >
              weld_row_offsets[std::size_t(plane)])
            out.push_back(plane);
        },
        tf::checked);
  // A retirement reaching a row CHANGES the group that row states, so the
  // group goes local with every carrier of it. A row still the world's names
  // the group the port must take; one already local names a group taken by
  // the wave that changed it.
  tf::generic_generate(
      tf::make_range(weld_row_data), taken,
      [n_immutable_canon](Index ticket, tf::buffer<Index> &out) {
        if (ticket < n_immutable_canon)
          out.push_back(ticket);
      },
      tf::checked);
  if (split_edge.size() != 0) {
    // an edge is an edge: the one lookup answers its current span — a local
    // id directly, an original one through the router
    const auto routed_span = [&](std::size_t at) {
      const auto group = split_edge[at];
      if (split_tier[at] == Index(1))
        return local_tables.canon_group(group);
      const auto local = routed_group(group);
      return local == Index(-1) ? world_tables.canon_group(group)
                                : local_tables.canon_group(local);
    };
    const auto index_extent = std::size_t(std::numeric_limits<Index>::max());
    tf::buffer<decltype(routed_span(std::size_t(0)))> spans;
    spans.allocate(split_edge.size());
    tf::buffer<Index> span_prefix;
    span_prefix.allocate(split_edge.size() + 1);
    span_prefix[0] = Index(0);
    for (std::size_t at = 0; at < split_edge.size(); ++at) {
      spans[at] = routed_span(at);
      const auto count = spans[at].size();
      if (count > index_extent - std::size_t(span_prefix[at]))
        return false;
      span_prefix[at + 1] = span_prefix[at] + Index(count);
    }
    const auto plane_base = planes.size();
    planes.reallocate(plane_base + std::size_t(span_prefix[split_edge.size()]));
    tf::parallel_for_each(
        tf::make_sequence_range(split_edge.size()), [&](std::size_t at) {
          auto row = plane_base + std::size_t(span_prefix[at]);
          for (const auto &def : spans[at])
            planes[row++] = plane_of_face(def.face);
        });
  }
  // a split states its parent's whole span, so an unrouted parent is a group
  // this wave takes
  tf::generic_generate(
      tf::make_sequence_range(split_edge.size()), taken,
      [&](std::size_t at, tf::buffer<Index> &out) {
        if (split_tier[at] == Index(0) &&
            routed_group(split_edge[at]) == Index(-1))
          out.push_back(split_edge[at]);
      },
      tf::checked);
  if (taken.size() != 0) {
    tbb::parallel_sort(taken.begin(), taken.end());
    taken.erase_till_end(std::unique(taken.begin(), taken.end()));
  }
  if (planes.size() == 0)
    return true;
  tbb::parallel_sort(planes.begin(), planes.end());
  planes.erase_till_end(std::unique(planes.begin(), planes.end()));
  return true;
}

} // namespace tf::arrangement
