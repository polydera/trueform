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
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./close_plane_carrier_frontier.hpp"
#include "./make_plane_generation_source.hpp"
#include "./plane_refinement_plan.hpp"

#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// Build refinement's initial sparse PA-local source.
///
/// The selected immutable planes close under every group they name; newly
/// reached source faces occupy the contiguous suffix and are local from birth.
/// One canonical record sort therefore fuses equal keys across both sources,
/// after which `plane_ticket` and `group_router` are the ordinary PA answers.
template <typename Index, typename Int, typename World, typename Splits,
          typename VertexOffsets>
auto make_plane_refinement_overlay(
    const World &world, const Splits &splits,
    const tf::intersect::graph::plane_tables<Index, Int> &entrants,
    const tf::buffer<Index> &entrant_origins,
    const VertexOffsets &vertex_offsets,
    tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    tf::buffer<Index> &plane_ticket, tf::buffer<Index> &group_router,
    tf::buffer<Index> &immutable_group_ticket,
    tf::buffer<Index> &entrant_group_ticket) -> bool {
  const auto entrant_count = Index(entrants.edges().size());
  const auto plane_extent = world.n_planes() + entrant_count;
  if (entrant_origins.size() != std::size_t(entrants.n_canon()))
    return false;

  if (tf::parallel_contains(
          splits,
          [&world](const auto &split) {
            return split.source != plane_refinement_immutable_source ||
                   split.group < Index(0) || split.group >= world.n_canon();
          },
          tf::checked) ||
      tf::parallel_contains(
          entrant_origins, [](Index origin) { return origin < Index(0); },
          tf::checked))
    return false;
  tf::buffer<Index> immutable_planes;
  tf::generic_generate(
      splits, immutable_planes,
      [&world](const auto &split, tf::buffer<Index> &out) {
        for (const auto &def : world.canon_group(split.group))
          out.push_back(world.plane_of_face(def.face));
      },
      tf::checked);
  tf::generic_generate(
      entrant_origins, immutable_planes,
      [&world](Index origin, tf::buffer<Index> &out) {
        if (origin >= world.n_canon())
          return;
        for (const auto &def : world.canon_group(origin))
          out.push_back(world.plane_of_face(def.face));
      },
      tf::checked);
  if (immutable_planes.size() != 0) {
    tbb::parallel_sort(immutable_planes.begin(), immutable_planes.end());
    immutable_planes.erase_till_end(
        std::unique(immutable_planes.begin(), immutable_planes.end()));
    const tf::buffer<Index> no_tickets;
    const auto plane_of_face = [&world](Index face) {
      return world.plane_of_face(face);
    };
    close_plane_carrier_frontier(world.tables(), no_tickets, plane_of_face,
                                 immutable_planes);
  }

  tf::buffer<Index> entrant_global_planes;
  entrant_global_planes.allocate(std::size_t(entrant_count));
  tf::parallel_iota(entrant_global_planes, world.n_planes());
  tf::buffer<Index> entrant_plane_ticket;
  if (!make_plane_generation_ticket(entrant_global_planes, plane_extent,
                                    entrant_plane_ticket))
    return false;

  tf::buffer<Index> entrant_origin_offsets;
  entrant_origin_offsets.allocate(std::size_t(entrants.n_canon()) + 1);
  tf::parallel_iota(entrant_origin_offsets, Index(0));

  tf::buffer<Index> global_planes;
  global_planes.reserve(immutable_planes.size() + entrant_global_planes.size());
  tf::core::append(immutable_planes, global_planes);
  tf::core::append(entrant_global_planes, global_planes);

  tf::buffer<Index> local_plane_of_global;
  tf::buffer<Index> local_origin_offsets;
  tf::buffer<Index> local_origins;
  const tf::buffer<std::array<Index, 3>> no_merges;
  if (!make_plane_generation_source<Index, Int>(
          world, entrants, entrant_plane_ticket, entrant_origin_offsets,
          entrant_origins, global_planes, plane_extent, no_merges,
          vertex_offsets, local_tables, local_plane_of_global,
          local_origin_offsets, local_origins, immutable_group_ticket,
          entrant_group_ticket) ||
      !make_plane_generation_ticket(local_plane_of_global, plane_extent,
                                    plane_ticket))
    return false;

  group_router = immutable_group_ticket;
  if (tf::parallel_contains(
          tf::make_sequence_range(world.n_canon()),
          [&](Index group) {
            const auto local = group_router[std::size_t(group)];
            if (local == Index(-1))
              return false;
            if (local < Index(0) || local >= local_tables.n_canon())
              return true;
            for (const auto &def : world.canon_group(group)) {
              const auto plane = world.plane_of_face(def.face);
              if (plane < Index(0) || plane >= world.n_planes() ||
                  plane_ticket[std::size_t(plane)] == Index(-1))
                return true;
            }
            return false;
          },
          tf::checked))
    return false;
  return true;
}

} // namespace tf::arrangement
