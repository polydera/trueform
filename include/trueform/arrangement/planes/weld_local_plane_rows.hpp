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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_def_respan.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_piece_key.hpp"
#include "./plane_tier_definitions.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

namespace tf::arrangement {

/// Substitute the wave's merged identities into the local rows that name them.
///
/// A weld is an identity substitution: the merge table states which identity
/// speaks for which, so a touched row's endpoints are rewritten through
/// @ref tf::intersect::graph::plane_merged_endpoint and the record restated
/// by @ref tf::intersect::graph::plane_piece_def — the local arrangement's
/// own producers — which is why the reversed and radial bits follow the new
/// key order exactly as a piece's do, and why the whole-side claim survives:
/// a weld moves an edge's endpoints, it does not cut the edge.
///
/// THE KEY IS THE GROUP'S. The substitution is therefore decided once, on the
/// group's first instance, and applied to its whole span, and the two states
/// it can leave a group in are solved on their own structures:
///
/// - a SURVIVOR keeps its identity, its instances and its triangulations. Its
///   carriers' blocks are ordered again — the key moved, and key order is the
///   invariant every block consumer indexes by — but the constraint set is
///   the same set of edges, so the plane does NOT re-triangulate and its
///   corners are remapped through this same merge table by
///   @ref tf::arrangement::rewrite_plane_flat_triangles. A surviving weld
///   publishes no plane in `recdt_planes`.
/// - a DEGENERATE group is no 1-cell at all: both endpoints became one
///   identity. It retires with no successor — every router entry naming it
///   answers `-2`, and its span stays where it is, dead history — and every
///   carrier's block is rebuilt without its rows and repointed. A WELD ALONE
///   NEVER RE-TRIANGULATES A RETAINED TRIANGULATION, zero-length occurrence or
///   not: the projection would state the same coincidence again, so the
///   retained triangulation stays, its corners rewrite through the closed
///   table, and its collapsed triangles compact at the boundary.
///   `recdt_planes` is always empty here — this operation states identities,
///   never a constraint set. A carrier that RETAINED NOTHING is the wave's
///   own question, and @ref tf::arrangement::advance_plane_wave asks it: a
///   plane whose constraints intersected refused, and the substitution that
///   retired the collision is the set it must read again.
///
/// `changed_groups` is the survivors, ascending: a touched row names an
/// endpoint this table moves, so their keys moved, and the wave's canonicalize
/// closes them under equal key beside its split children. This operation fuses
/// nothing — two groups may leave it holding one key, which is what a weld
/// means and what the canonicalize is for.
///
/// The rows are the gather's, per plane, stated in the flat row space the
/// sweep read them in. A retirement reaching a row CHANGES the group that row
/// states, so the wave TOOK that group before this runs: a row still the
/// world's names its taken group through the router, and that is the whole
/// correspondence — no second map exists and none may be introduced. The
/// carriers are read off the spans instead, because a group's instances name
/// them whether or not the sweep listed a row there, and every one of them
/// holds a local block or the call refuses.
///
/// False rejects the call and publishes nothing: the local tier's interior was
/// proven by the producers that appended it, so this states only what the
/// substitution itself indexes.
template <typename Index, typename Int, typename VertexOffsets,
          typename PlaneOfFace>
auto weld_local_plane_rows(
    const tf::intersect::graph::plane_tables<Index, Int> &world_tables,
    const tf::buffer<std::array<Index, 3>> &merges,
    const VertexOffsets &vertex_offsets,
    const tf::buffer<Index> &weld_row_offsets,
    const tf::buffer<Index> &weld_row_data, const PlaneOfFace &plane_of_face,
    tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    tf::buffer<Index> &plane_ticket, tf::buffer<Index> &group_router,
    tf::buffer<Index> &recdt_planes, tf::buffer<Index> &changed_groups)
    -> bool {
  const auto n_canon = local_tables.n_canon();
  const auto n_blocks = local_tables.edges().size();
  if (!local_tables.well_formed())
    return false;
  if (weld_row_offsets.size() == 0) {
    if (weld_row_data.size() != 0)
      return false;
  } else if (weld_row_offsets.size() != plane_ticket.size() + 1 ||
             weld_row_offsets[0] != Index(0) ||
             weld_row_offsets[weld_row_offsets.size() - 1] !=
                 Index(weld_row_data.size()))
    return false;
  // a plane the sweep found weld rows in states them in the LOCAL tier, so it
  // holds a local block by the time the surgery runs
  if (tf::parallel_contains(
          tf::make_sequence_range(plane_ticket.size()),
          [&](std::size_t plane) {
            const auto block = plane_ticket[plane];
            if (block < Index(-1) || block >= Index(n_blocks))
              return true;
            return weld_row_offsets.size() != 0 &&
                   (weld_row_offsets[plane] > weld_row_offsets[plane + 1] ||
                    (weld_row_offsets[plane] < weld_row_offsets[plane + 1] &&
                     block == Index(-1)));
          },
          tf::checked))
    return false;
  const auto tier =
      make_plane_tier_definitions(world_tables, local_tables, true);
  const auto &def_offsets = local_tables.def_offsets();
  const auto canon_base = world_tables.n_canon();
  // the group a TICKET states NOW: a ticket the world still owned when the
  // sweep read it answers through the router, because a retirement reaching a
  // row is exactly what makes the wave take that group
  const auto group_of_ticket = [&](Index ticket) {
    return ticket < canon_base ? group_router[std::size_t(ticket)]
                               : ticket - canon_base;
  };
  if (group_router.size() != std::size_t(canon_base) ||
      tf::parallel_contains(
          tf::make_range(weld_row_data),
          [&](Index ticket) {
            if (ticket < Index(0))
              return true;
            const auto group = group_of_ticket(ticket);
            return group < Index(0) || group >= n_canon ||
                   def_offsets[std::size_t(group) + 1] <=
                       def_offsets[std::size_t(group)];
          },
          tf::checked))
    return false;
  if (tf::parallel_contains(
          tf::make_range(group_router),
          [n_canon](Index group) {
            return group < Index(-2) || group >= n_canon;
          },
          tf::checked))
    return false;
  if (weld_row_data.size() == 0) {
    recdt_planes.clear();
    changed_groups.clear();
    return true;
  }

  // a weld touches a group WHOLE: every instance states the same key, so the
  // rows name groups and the groups are the grain
  tf::buffer<Index> touched;
  touched.allocate(weld_row_data.size());
  tf::parallel_transform(tf::make_range(weld_row_data), touched,
                         group_of_ticket, tf::checked);
  tbb::parallel_sort(touched.begin(), touched.end());
  touched.erase_till_end(std::unique(touched.begin(), touched.end()));

  // the substitution one group states: its endpoints as the table leaves them
  struct weld_group_t {
    std::array<Index, 2> from;
    std::array<Index, 2> to;
    Index group;
  };
  tf::buffer<weld_group_t> states;
  states.allocate(touched.size());
  tf::parallel_for_each(
      tf::make_sequence_range(touched.size()),
      [&](std::size_t at) {
        const auto &def = local_tables.canon_group(touched[at])[0];
        const auto ends = tf::intersect::graph::plane_merged_endpoints(
            merges, vertex_offsets, def);
        states[at] = {{Index(ends.lo_tag), ends.lo_id},
                      {Index(ends.hi_tag), ends.hi_id},
                      touched[at]};
      },
      tf::checked);

  tf::buffer<Index> retired_groups;
  tf::sequenced_generate(
      tf::make_range(states), retired_groups,
      [](const weld_group_t &state, tf::buffer<Index> &out) {
        if (state.from == state.to)
          out.push_back(state.group);
      },
      tf::checked);
  tf::buffer<weld_group_t> survivors;
  tf::sequenced_generate(
      tf::make_range(states), survivors,
      [](const weld_group_t &state, tf::buffer<weld_group_t> &out) {
        if (state.from != state.to)
          out.push_back(state);
      },
      tf::checked);

  tf::buffer<Index> dead_planes;
  tf::generic_generate(tf::make_range(retired_groups), dead_planes,
                       [&](Index group, tf::buffer<Index> &out) {
                         for (const auto &def : local_tables.canon_group(group))
                           out.push_back(plane_of_face(def.face));
                       },
                       tf::checked);
  if (dead_planes.size() != 0) {
    tbb::parallel_sort(dead_planes.begin(), dead_planes.end());
    dead_planes.erase_till_end(
        std::unique(dead_planes.begin(), dead_planes.end()));
  }
  tf::buffer<Index> weld_planes;
  tf::generic_generate(tf::make_range(survivors), weld_planes,
                       [&](const weld_group_t &state, tf::buffer<Index> &out) {
                         for (const auto &def :
                              local_tables.canon_group(state.group))
                           out.push_back(plane_of_face(def.face));
                       },
                       tf::checked);
  if (weld_planes.size() != 0) {
    tbb::parallel_sort(weld_planes.begin(), weld_planes.end());
    weld_planes.erase_till_end(
        std::unique(weld_planes.begin(), weld_planes.end()));
  }
  const auto unported = [&](Index plane) {
    if (plane < Index(0) || std::size_t(plane) >= plane_ticket.size())
      return true;
    const auto block = plane_ticket[std::size_t(plane)];
    return block < Index(0) || std::size_t(block) >= n_blocks;
  };
  if (tf::parallel_contains(tf::make_range(dead_planes), unported,
                            tf::checked) ||
      tf::parallel_contains(tf::make_range(weld_planes), unported,
                            tf::checked))
    return false;
  // ONE OWNER PER BLOCK: a carrier of a retired group is rebuilt, and only the
  // carriers no rebuild reaches are ordered in place
  tf::buffer<Index> ordered_planes;
  tf::sequenced_generate(
      tf::make_range(weld_planes), ordered_planes,
      [&dead_planes](Index plane, tf::buffer<Index> &out) {
        if (!std::binary_search(dead_planes.begin(), dead_planes.end(), plane))
          out.push_back(plane);
      },
      tf::checked);

  auto &block_offsets = local_tables.edges().offsets_buffer();
  auto &block_rows = local_tables.edges().data_buffer();
  const auto retired_group = [&retired_groups](Index group) {
    return std::binary_search(retired_groups.begin(), retired_groups.end(),
                              group);
  };
  // a row still the world's states a group the wave never changed, so no
  // retirement can reach it
  const auto retired_row = [&](Index row) {
    return !tier.immutable(row) && retired_group(tier[std::size_t(row)].id);
  };
  tf::buffer<std::size_t> row_prefix;
  row_prefix.allocate(dead_planes.size() + 1);
  row_prefix[0] = 0;
  tf::parallel_for_each(
      tf::make_sequence_range(dead_planes.size()), [&](std::size_t at) {
        std::size_t count = 0;
        for (const auto row : local_tables.plane_edges(
                 plane_ticket[std::size_t(dead_planes[at])]))
          count += std::size_t(!retired_row(row));
        row_prefix[at + 1] = count;
      });
  // every kept row is a row the tier already holds, so the sum is bounded by
  // the CSR itself and only the appended extents can exceed the index space
  for (std::size_t at = 0; at < dead_planes.size(); ++at)
    row_prefix[at + 1] += row_prefix[at];
  const auto index_extent = std::size_t(std::numeric_limits<Index>::max());
  if (dead_planes.size() > index_extent - n_blocks ||
      row_prefix[dead_planes.size()] > index_extent - block_rows.size())
    return false;

  auto &table_defs = local_tables.defs();
  tf::parallel_for_each(
      tf::make_range(survivors), [&](const weld_group_t &state) {
        for (auto row = def_offsets[std::size_t(state.group)];
             row < def_offsets[std::size_t(state.group) + 1]; ++row) {
          auto &def = table_defs[std::size_t(row)];
          const auto parent = def;
          tf::intersect::graph::plane_piece_def(def, state.from, state.to,
                                                 parent, Index(1));
        }
      });

  struct block_local_t {
    tf::buffer<std::array<Index, 5>> keyed;
  };
  tf::parallel_for_each(
      tf::make_range(ordered_planes),
      [&](Index plane, block_local_t &local) {
        const auto block = std::size_t(plane_ticket[std::size_t(plane)]);
        sort_plane_block_by_key(tier, block_rows,
                                std::size_t(block_offsets[block]),
                                std::size_t(block_offsets[block + 1]),
                                local.keyed);
      },
      block_local_t{});

  if (dead_planes.size() != 0) {
    const auto block_base = Index(n_blocks);
    const auto row_base = std::size_t(block_rows.size());
    const auto old_block_offsets = block_offsets.size();
    block_offsets.reallocate(old_block_offsets + dead_planes.size());
    block_rows.reallocate(row_base + row_prefix[dead_planes.size()]);
    tf::parallel_for_each(
        tf::make_sequence_range(dead_planes.size()),
        [&](std::size_t at, block_local_t &local) {
          const auto plane = dead_planes[at];
          const auto block = std::size_t(plane_ticket[std::size_t(plane)]);
          const auto begin = row_base + row_prefix[at];
          auto write = begin;
          for (auto source = std::size_t(block_offsets[block]);
               source < std::size_t(block_offsets[block + 1]); ++source) {
            const auto row = block_rows[source];
            if (!retired_row(row))
              block_rows[write++] = row;
          }
          sort_plane_block_by_key(tier, block_rows, begin, write, local.keyed);
          block_offsets[old_block_offsets + at] = Index(write);
          plane_ticket[std::size_t(plane)] = block_base + Index(at);
        },
        block_local_t{});
  }

  if (retired_groups.size() != 0)
    tf::parallel_for_each(
        tf::make_sequence_range(group_router.size()),
        [&](std::size_t root) {
          const auto group = group_router[root];
          if (group >= Index(0) && retired_group(group))
            group_router[root] = Index(-2);
        },
        tf::checked);

  changed_groups.clear();
  changed_groups.allocate(survivors.size());
  tf::parallel_transform(
      tf::make_range(survivors), changed_groups,
      [](const weld_group_t &state) { return state.group; }, tf::checked);
  return true;
}

} // namespace tf::arrangement
