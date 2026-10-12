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
#include "../../core/reallocate.hpp"
#include "../../intersect/graph/flat_of_vertex.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_carrier_boundary.hpp"
#include "./plane_definition_source.hpp"
#include "./plane_member_statements.hpp"
#include "./plane_tier_definitions.hpp"
#include "./prepare_plane_carrier_boundary.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// One plane's triangulation input, read straight off the prepared
/// records: the distinct endpoints of its edge block are its point
/// table (a sort, not a map), every definition is one constraint, and
/// a boundary definition states its own member — so a stack's parity
/// has the multiplicity the instances give it.
///
/// A world that has not materialized its definition tier holds no block to
/// read: it has stated nothing about a carrier but the face it is, and
/// @ref tf::arrangement::prepare_plane_carrier_boundary states the same input
/// off that face. A world that never defers one never compiles the branch.
template <typename Index, typename Int, typename World, typename VertexOffsets,
          typename Local, typename PointOfFlat>
auto prepare_plane_triangulation(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket, const VertexOffsets &vertex_offsets,
    Index plane, Local &local, const PointOfFlat &point_of_flat) -> bool {
  if constexpr (states_carrier_boundary<World, Index>) {
    if (!world.materialized())
      return prepare_plane_carrier_boundary(world, vertex_offsets, plane, local,
                                            point_of_flat);
  }
  const auto source =
      find_plane_definition_source(world, local_tables, plane_ticket, plane);
  const auto plane_block = source.tables.plane_edges(source.block);
  // a run is one TICKET: a group still the world's and one this arrangement
  // owns never merge, whichever tier the block itself belongs to
  const auto tier = make_plane_tier_definitions(world.tables(), local_tables,
                                                !source.immutable);
  const auto n_members = std::size_t(world.member_count(plane));
  const bool pooled = n_members > 1;
  if (plane_block.size() == 0)
    return false;

  const auto flat_of = [&](auto tag, Index id) {
    return tf::intersect::graph::flat_of_vertex(vertex_offsets, tag, id);
  };
  local.ends.clear();
  tf::core::reallocate(local.ends, plane_block.size() * 2);
  auto *write = local.ends.begin();
  for (const auto e : plane_block) {
    const auto &def = tier[std::size_t(e)];
    *write++ = flat_of(def.point_tag_0, def.point_0);
    *write++ = flat_of(def.point_tag_1, def.point_1);
  }
  std::sort(local.ends.begin(), local.ends.end());
  local.ends.erase_till_end(std::unique(local.ends.begin(), local.ends.end()));
  auto local_of = [&](Index flat) -> Index {
    return Index(std::lower_bound(local.ends.begin(), local.ends.end(), flat) -
                 local.ends.begin());
  };

  const auto &frame = world.frame(plane);
  local.pts2.clear();
  local.pts2.reserve(local.ends.size());
  for (const auto flat : local.ends) {
    const auto q = point_of_flat(flat);
    local.pts2.emplace_back(q[frame.ax0], q[frame.ax1]);
  }

  local.face_orientation =
      int(world.face_orientation(world.member(plane, Index(0))));

  // an instance names its face, so the members are stated once as a sorted
  // (face, ordinal) table and every instance answers by one search
  local.member_ticket.clear();
  if (pooled) {
    tf::core::reallocate(local.member_ticket, n_members);
    for (std::size_t m = 0; m < n_members; ++m)
      local.member_ticket[m] = {world.member(plane, Index(m)), Index(m)};
    std::sort(local.member_ticket.begin(), local.member_ticket.end());
  }
  const auto member_of = [&local, n_members](Index face) {
    const auto at =
        std::lower_bound(local.member_ticket.begin(),
                         local.member_ticket.end(),
                         std::array<Index, 2>{face, Index(0)});
    return at != local.member_ticket.end() && (*at)[0] == face
               ? std::size_t((*at)[1])
               : n_members;
  };

  // The block is canon-grouped, so the unique edges are a run walk and
  // the fold IS the attribution: a canonical edge is a sheet boundary
  // when some member's own side states it, and the members that do are
  // exactly the ones the coverage flood crosses it for. Coincident
  // sides pool into ONE group, so the winding carries the multiplicity
  // a per-instance statement would lose to the triangulation's own
  // deduplication of coincident constraints. The same walk states the
  // original side the run lies on for each member, which is everything
  // emission needs to stamp its edges and its corners.
  local.cons.clear();
  local.bnd.clear();
  local.cons_group.clear();
  local.cons_row.clear();
  local.cons_statements.clear();
  local.cons_side.clear();
  local.cons_winding.clear();
  local.cons_cut.clear();
  for (std::size_t i = 0; i < plane_block.size();) {
    const auto &def = tier[std::size_t(plane_block[i])];
    const auto ticket = tier.ticket(plane_block[i]);
    std::size_t j = i;
    const auto row = local.cons_statements.size();
    const auto cuts = local.cons_cut.size();
    bool boundary = false;
    short side = -1;
    // a member's statements ADD their directions: a member stating the
    // group both ways is the mesh folding onto one carrier, and its wall
    // cancels — the repeated-boundary law the triangulation itself states,
    // which the run collapse would otherwise lose
    int winding = 0;
    while (j < plane_block.size() &&
           tier.ticket(plane_block[j]) == ticket) {
      const auto &instance = tier[std::size_t(plane_block[j])];
      if (instance.ordinal >= 0) {
        const auto direction = static_cast<signed char>(
            instance.flags & tf::intersect::graph::plane_edge_reversed_flag
                ? -1
                : 1);
        if (pooled) {
          const auto m = member_of(instance.face);
          if (m != n_members)
            state_plane_member_statement(local.cons_statements, row,
                                         Index(local.bnd.size()), Index(m),
                                         instance.side, direction);
        } else {
          winding += direction;
          if (side < short(0))
            side = instance.side;
        }
      } else {
        const auto m = pooled ? member_of(instance.face) : std::size_t(0);
        if (m != n_members)
          local.cons_cut.push_back(
              {Index(local.bnd.size()), Index(m)});
      }
      ++j;
    }
    i = j;
    // the members stating this cut, once each and ascending
    std::sort(local.cons_cut.begin() + std::ptrdiff_t(cuts),
              local.cons_cut.end());
    local.cons_cut.erase_till_end(
        std::unique(local.cons_cut.begin() + std::ptrdiff_t(cuts),
                    local.cons_cut.end()));
    if (pooled) {
      for (auto at = row; at < local.cons_statements.size(); ++at)
        boundary = boundary || (local.cons_statements[at].winding & 1) != 0;
    } else
      boundary = (winding & 1) != 0;
    const auto a = local_of(flat_of(def.point_tag_0, def.point_0));
    const auto b = local_of(flat_of(def.point_tag_1, def.point_1));
    const auto pa = local.pts2[std::size_t(a)];
    const auto pb = local.pts2[std::size_t(b)];
    if (a == b || (pa[0] == pb[0] && pa[1] == pb[1])) {
      if (pooled)
        local.cons_statements.erase_till_end(local.cons_statements.begin() +
                                             std::ptrdiff_t(row));
      local.cons_cut.erase_till_end(local.cons_cut.begin() +
                                    std::ptrdiff_t(cuts));
      continue;
    }
    local.cons.push_back(a);
    local.cons.push_back(b);
    local.bnd.push_back(char(boundary));
    local.cons_group.push_back(ticket);
    if (pooled)
      local.cons_row.push_back(
          {Index(row), Index(local.cons_statements.size())});
    else {
      local.cons_side.push_back(side);
      local.cons_winding.push_back(static_cast<signed char>(winding));
    }
  }
  // the row an edge belonging to no constraint reads: nobody states it
  if (pooled)
    local.cons_row.push_back({Index(0), Index(0)});
  return true;
}

} // namespace tf::arrangement
