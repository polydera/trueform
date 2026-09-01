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

#include "../../core/reallocate.hpp"
#include "../../intersect/graph/flat_of_vertex.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// CORE. One carrier's triangulation input, read off THE FACE IT IS.
///
/// A world that has not materialized its definition tier has stated nothing
/// about a carrier but its own boundary, so the face's corners ARE the
/// constraint set — and the loop names each identity once, so the loop IS the
/// point table, its position IS the local index, and its edges are consecutive
/// positions. No block, no sort, no unique, no search: the same law
/// @ref tf::iso::prepare_iso_face_cut states for a face's own chain.
///
/// A corner an input repeats is one identity twice, and a point table names
/// each once, so the walk keeps the first position of each — a face's corners
/// are a tiny bounded population, so the scan that finds it is the whole
/// mechanism, here and for the fold below. The side such a corner closes then
/// bounds nothing and states no constraint, exactly as a block of zero length
/// would.
template <typename Index, typename World, typename VertexOffsets,
          typename Local, typename PointOfFlat>
auto prepare_plane_carrier_boundary(const World &world,
                                    const VertexOffsets &vertex_offsets,
                                    Index plane, Local &local,
                                    const PointOfFlat &point_of_flat) -> bool {
  const auto corners = world.carrier_boundary(plane);
  const auto n_corners = corners.size();
  if (n_corners < 3)
    return false;

  local.ends.clear();
  local.corner_local.clear();
  tf::core::reallocate(local.corner_local, n_corners);
  for (std::size_t corner = 0; corner < n_corners; ++corner) {
    const auto flat = tf::intersect::graph::flat_of_vertex(
        vertex_offsets, std::int16_t(0), Index(corners[corner]));
    auto position = Index(local.ends.size());
    for (std::size_t at = 0; at < local.ends.size(); ++at)
      if (local.ends[at] == flat) {
        position = Index(at);
        break;
      }
    if (position == Index(local.ends.size()))
      local.ends.push_back(flat);
    local.corner_local[corner] = position;
  }

  const auto &frame = world.frame(plane);
  local.pts2.clear();
  local.pts2.reserve(local.ends.size());
  for (const auto flat : local.ends) {
    const auto q = point_of_flat(flat);
    local.pts2.emplace_back(q[frame.ax0], q[frame.ax1]);
  }
  local.face_orientation =
      int(world.face_orientation(world.member(plane, Index(0))));
  local.member_ticket.clear();
  local.cons_row.clear();

  // Every side is a boundary statement of the carrier's only member, so the
  // side index IS the ordinal. A SIDE STATED TWICE IS THE MESH FOLDING ONTO
  // ITSELF AND ITS WALL CANCELS — the repeated-boundary law a canon-major
  // block carries as a run's parity toggle, carried here as the toggle
  // itself, since equal pairs are what a canonical order would have made
  // adjacent. The group space is the tier's, and an unmaterialized tier
  // states none, so no slot of this carrier names a piece.
  local.cons.clear();
  local.bnd.clear();
  local.cons_group.clear();
  local.cons_side.clear();
  for (std::size_t side = 0; side < n_corners; ++side) {
    const auto a = local.corner_local[side];
    const auto b = local.corner_local[(side + 1) % n_corners];
    const auto pa = local.pts2[std::size_t(a)];
    const auto pb = local.pts2[std::size_t(b)];
    if (a == b || (pa[0] == pb[0] && pa[1] == pb[1]))
      continue;
    auto stated = local.bnd.size();
    for (std::size_t at = 0; at < local.bnd.size(); ++at) {
      const auto x = local.cons[2 * at];
      const auto y = local.cons[2 * at + 1];
      if ((x == a && y == b) || (x == b && y == a)) {
        stated = at;
        break;
      }
    }
    if (stated != local.bnd.size()) {
      local.bnd[stated] = char(local.bnd[stated] ^ 1);
      continue;
    }
    local.cons.push_back(a);
    local.cons.push_back(b);
    local.bnd.push_back(char(1));
    local.cons_group.push_back(Index(-1));
    local.cons_side.push_back(short(side));
  }
  return true;
}

} // namespace tf::arrangement
