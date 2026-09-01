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
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "../../topology/label_connected_components.hpp"
#include "./plane_definition_source.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace tf::arrangement {

/// Close a GENERATION's plane selection under CARRIERS OF NAMED GROUPS.
///
/// A generation source rebuilds its whole tier out of the rows of the planes
/// it is given, so a group's span there is complete only when every carrier of
/// it was selected — and a span that is not complete is not the group. That is
/// the one closure this arrangement still owes: a seed takes with it the whole
/// COMPONENT of the still-immutable plane graph it sits in, where two planes
/// are adjacent exactly when one canonical group names them both.
///
/// A RECOVERY WAVE owes nothing of the kind. It ports blocks rather than
/// rebuilding a tier, and AN UNCHANGED GROUP STAYS THE WORLD'S, VERBATIM, FOR
/// BOTH CARRIERS, so its frontier is the changed groups' own carriers — a ring,
/// closed by construction where those groups are named.
///
/// A component is a fact of the world's own adjacency, not of the wave, so it
/// is one flood at PLANE grain: an immutable plane's neighbours are the
/// carriers of the groups its block names, and the seeds' labels then select
/// the planes that join. The labels are total before the finder runs and its
/// walk reaches every candidate, so a component's DIAMETER never enters the
/// cost.
///
/// Joining is not retriangulation: a plane keeps whatever product it holds, a
/// refusal included — the rows move verbatim, so nothing it would read has
/// changed.
template <typename Index, typename Int, typename PlaneOfFace>
auto close_plane_carrier_frontier(
    const tf::intersect::graph::plane_tables<Index, Int> &world_tables,
    const tf::buffer<Index> &plane_ticket, const PlaneOfFace &plane_of_face,
    tf::buffer<Index> &frontier) -> void {
  if (frontier.size() == 0)
    return;
  const auto world_defs = world_tables.edge_defs();
  const auto n_planes = plane_ticket_space(world_tables, plane_ticket);
  tf::buffer<char> port_candidate;
  port_candidate.allocate(n_planes);
  tf::parallel_for_each(
      tf::make_sequence_range(Index(n_planes)),
      [&](Index plane) {
        port_candidate[std::size_t(plane)] =
            plane_reads_world_tier(plane_ticket, plane) ? char(1) : char(0);
      },
      tf::checked);
  // the finder names the planes its walk reached and leaves the rest as it
  // found them, so the buffer is total before it runs
  tf::buffer<Index> plane_label;
  plane_label.allocate(n_planes);
  tf::parallel_fill(plane_label, Index(-1));
  const auto applier = [&](Index plane, const auto &push) {
    for (const auto row : world_tables.plane_edges(plane))
      for (const auto &def :
           world_tables.canon_group(world_defs[std::size_t(row)].id))
        push(plane_of_face(def.face));
  };
  const auto n_components = tf::label_connected_components_masked<Index>(
      plane_label, port_candidate, applier);

  tf::buffer<char> seeded;
  seeded.allocate_and_initialize(std::size_t(n_components), char(0));
  tf::parallel_for_each(
      frontier,
      [&](Index plane) {
        if (port_candidate[std::size_t(plane)] != char(0))
          seeded[std::size_t(plane_label[std::size_t(plane)])] = char(1);
      },
      tf::checked);
  tf::buffer<Index> closed;
  tf::sequenced_generate(
      tf::make_sequence_range(Index(n_planes)), closed,
      [&](Index plane, tf::buffer<Index> &out) {
        if (port_candidate[std::size_t(plane)] != char(0) &&
            seeded[std::size_t(plane_label[std::size_t(plane)])] != char(0))
          out.push_back(plane);
      },
      tf::checked);
  // a seed the flood never named — a plane already local — keeps the place in
  // the frontier the diff published it in
  tf::core::append(frontier, closed);
  tbb::parallel_sort(closed.begin(), closed.end());
  closed.erase_till_end(std::unique(closed.begin(), closed.end()));
  frontier = std::move(closed);
}

} // namespace tf::arrangement
