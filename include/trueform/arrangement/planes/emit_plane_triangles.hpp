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

#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../topology/cdt_refiner.hpp"
#include "../../topology/topo_id.hpp"
#include "../../topology/topo_type.hpp"
#include "./collect_plane_corner_welds.hpp"
#include "./elect_plane_survivors.hpp"
#include "./flood_plane_member_coverage.hpp"
#include "./gather_plane_member_triangles.hpp"
#include "./label_plane_cells.hpp"
#include "./plane_member_statements.hpp"
#include "./plane_orientation.hpp"
#include "./state_plane_member_point_subs.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::arrangement {

/// CORE. The piece a slot names is the constraint the plane's own table
/// states. Two inputs may cover one edge; the producers name different members
/// of that coincidence class, so the class's standing owner answers for all.
template <typename Index, typename Local, typename Mesh>
auto emit_plane_slots(const Local &local, const Mesh &mesh, Index triangle,
                      bool flip) -> std::array<Index, 3> {
  const auto owners = mesh.face_constraint_owners(triangle);
  std::array<Index, 3> slots{Index(-1), Index(-1), Index(-1)};
  for (int s = 0; s < 3; ++s) {
    const auto &owner = owners[std::size_t(flip ? 2 - s : s)];
    if (owner.input_id == Index(-1))
      continue;
    const auto input = local.cons_root.size() == 0
                           ? owner.input_id
                           : local.cons_root[std::size_t(owner.input_id)];
    slots[std::size_t(s)] = local.cons_group[std::size_t(input)];
  }
  return slots;
}

/// CORE. Where each corner of one emitted triangle sits on the emitting
/// member's own polygon, read off the point table the member's own sides
/// state.
template <typename Index, typename Local, typename Triangle>
auto emit_plane_subs(const Local &local, const Triangle &triangle)
    -> std::array<tf::topo_id<short>, 3> {
  std::array<tf::topo_id<short>, 3> subs;
  for (int c = 0; c < 3; ++c) {
    const auto representative = local.rep[std::size_t(triangle[std::size_t(c)])];
    subs[std::size_t(c)] =
        representative == Index(-1)
            ? tf::topo_id<short>{short(0), tf::topo_type::face}
            : local.point_subs[std::size_t(representative)];
  }
  return subs;
}

/// CORE. Every corner of the stock kernel's triangulation must be one of the
/// identities the plane was prepared with.
template <typename Index, typename Local>
auto plane_corners_are_named(const Local &local) -> bool {
  for (const auto representative : local.rep)
    if (representative == Index(-1))
      return false;
  return true;
}

/// CORE. A lone member's plane: everything the triangulation kept inside the
/// face boundary, in the face's own winding. Region 0 is the hull
/// exterior and the only thing dropped.
template <typename Index, typename Int, typename Local, typename Mesh,
          typename CornerOf>
auto emit_single_plane(Local &local, const Mesh &mesh,
                       const CornerOf &corner_of, bool record_arrangement,
                       bool record_cells) -> void {
  auto faces = mesh.make_faces();
  const bool flip =
      plane_cdt_orientation<Int>(faces, mesh.points()) != local.face_orientation;
  const auto labels = mesh.region_labels();
  const auto before = local.tris.size();
  state_plane_point_subs(local.cons, local.cons_side, local.ends.size(),
                         local.point_subs, local.subs_touched);
  for (std::size_t t = 0; t < std::size_t(faces.size()); ++t) {
    if (labels[t] == 0)
      continue;
    const auto triangle = faces[t];
    std::array<Index, 3> corners;
    for (int c = 0; c < 3; ++c)
      corners[std::size_t(c)] = corner_of(triangle[std::size_t(c)]);
    auto subs = emit_plane_subs<Index>(local, triangle);
    if (flip) {
      std::swap(corners[1], corners[2]);
      std::swap(subs[1], subs[2]);
    }
    if (record_arrangement) {
      local.slots.push_back(emit_plane_slots(local, mesh, Index(t), flip));
      local.coplanar_of.push_back(Index(-1));
      local.stacked.push_back(char(0));
    }
    local.subs.push_back(subs);
    local.tris.push_back(corners);
  }
  local.face_range.push_back({Index(before), Index(local.tris.size())});
  if (record_cells)
    record_single_plane_cells<Index>(local, mesh);
}

/// CORE. A coplanar stack: ONE triangulation, so agreement between members
/// is by construction and the only question is who covers what. Each
/// region elects the minimal-tag covering member; every other covering
/// member emits a duplicate pointing at the survivor.
template <typename Index, typename Int, typename World, typename Local,
          typename Mesh, typename CornerOf>
auto emit_plane_stack(const World &world, Index plane, Local &local,
                      const Mesh &mesh, const CornerOf &corner_of,
                      bool record_arrangement, bool record_cells) -> bool {
  auto faces = mesh.make_faces();
  const auto n_tri = std::size_t(faces.size());
  const auto members = world.plane_members(plane);
  const auto n_members = members.size();
  const auto n_regions = flood_plane_member_coverage<Index>(
      mesh, local.cons_row, local.cons_statements, Index(local.bnd.size()),
      local.region_adj, local.region_offsets, local.region_stack,
      local.region_reached, local.cover_range, local.cover);
  if (n_regions == Index(-1))
    return false;
  local.census.stack_regions += std::size_t(n_regions);
  const auto labels = mesh.region_labels();
  const int cdt_rev = plane_cdt_orientation<Int>(faces, mesh.points());
  local.member_rev.allocate(n_members);
  for (std::size_t mi = 0; mi < n_members; ++mi)
    local.member_rev[mi] =
        char(int(world.face_orientation(members[mi])) != cdt_rev);
  elect_plane_region_survivors(world, members, local.cover_range, local.cover,
                               n_regions, local.surv_mi);
  gather_plane_member_triangles(local, labels, n_tri, n_members,
                                Index(local.tris.size()), record_arrangement);
  order_plane_member_statements(local.cons_statements, local.cons_row,
                                local.bnd.size(), n_members,
                                local.statement_offsets, local.member_cursor,
                                local.member_statements);
  local.point_subs.allocate(local.ends.size());
  std::fill(local.point_subs.begin(), local.point_subs.end(),
            tf::topo_id<short>{short(0), tf::topo_type::face});
  local.subs_touched.clear();
  for (std::size_t mi = 0; mi < n_members; ++mi) {
    const auto before = local.tris.size();
    state_plane_member_point_subs(
        local.cons,
        tf::make_range(local.member_statements.begin() +
                           std::ptrdiff_t(local.statement_offsets[mi]),
                       local.member_statements.begin() +
                           std::ptrdiff_t(local.statement_offsets[mi + 1])),
        local.point_subs, local.subs_touched);
    for (auto k = local.member_offsets[mi]; k != local.member_offsets[mi + 1];
         ++k) {
      const auto t = std::size_t(local.member_tri[std::size_t(k)]);
      const auto triangle = faces[t];
      std::array<Index, 3> corners;
      for (int c = 0; c < 3; ++c)
        corners[std::size_t(c)] = corner_of(triangle[std::size_t(c)]);
      auto subs = emit_plane_subs<Index>(local, triangle);
      const bool flip = local.member_rev[mi] != 0;
      if (flip) {
        std::swap(corners[1], corners[2]);
        std::swap(subs[1], subs[2]);
      }
      local.subs.push_back(subs);
      const auto survivor = local.surv_mi[std::size_t(labels[t])];
      if (Index(mi) != survivor)
        ++local.census.dead_triangles;
      if (record_arrangement) {
        local.slots.push_back(emit_plane_slots(local, mesh, Index(t), flip));
        const auto covering = local.cover_range[std::size_t(labels[t])];
        local.stacked.push_back(char(covering[1] - covering[0] > Index(1)));
        if (Index(mi) == survivor)
          local.coplanar_of.push_back(Index(-1));
        else {
          local.coplanar_of.push_back(Index(local.coplanar.size()));
          local.coplanar.push_back(
              {local.surv_pos[t],
               char(local.member_rev[mi] !=
                    local.member_rev[std::size_t(survivor)])});
        }
      }
      local.tris.push_back(corners);
    }
    local.face_range.push_back({Index(before), Index(local.tris.size())});
  }
  if (record_cells)
    record_stack_plane_cells<Index>(local, mesh);
  return true;
}

/// CORE. Emit one prepared plane from the triangulation that built it. A
/// refined mesh carries interior points no input names: each takes a
/// PLANE-LOCAL ticket at its first corner and states its exact position once.
/// The plane is the identity's carrier, so a later round that rebuilds the
/// plane replaces both, and materialization names what the arena finally
/// holds.
///
/// THE ARRANGEMENT FACTS A TRIANGLE CARRIES — the piece each constrained slot
/// lies on, the survivor a coincident duplicate names, and whether its region
/// is stacked — are stated only for a build that asked for them. They are
/// facts about how a triangle sits among MANY carriers, so a caller that never
/// reads them pays neither the constraint-owner walk nor the store.
template <typename Index, typename Int, typename World, typename Local,
          typename Mesh>
auto emit_plane_triangles(const World &world, Index plane, Index n_flat,
                          Local &local, const Mesh &mesh, bool pooled,
                          bool record_arrangement, bool record_cells) -> bool {
  if (!collect_plane_corner_welds(local, mesh, plane, n_flat))
    return false;
  const auto steiner_base = Index(local.steiner_points.size());
  if constexpr (std::is_same_v<std::decay_t<Mesh>,
                               tf::cdt_refiner<Index, Int, Int>>) {
    local.steiner_ticket.allocate(std::size_t(mesh.points().size()));
    std::fill(local.steiner_ticket.begin(), local.steiner_ticket.end(),
              Index(-1));
  } else if (!plane_corners_are_named<Index>(local))
    return false;
  const auto corner_of = [&](Index point) -> Index {
    const auto representative = local.rep[std::size_t(point)];
    if constexpr (std::is_same_v<std::decay_t<Mesh>,
                                 tf::cdt_refiner<Index, Int, Int>>) {
      if (representative == Index(-1)) {
        auto &ticket = local.steiner_ticket[std::size_t(point)];
        if (ticket == Index(-1)) {
          ticket = Index(local.steiner_points.size()) - steiner_base;
          const auto &site = mesh.points()[std::size_t(point)];
          local.steiner_points.push_back(tf::exact::lift_plane_point(
              world.frame(plane), tf::make_point(site[0], site[1])));
        }
        return Index(-2) - ticket;
      }
    }
    return local.ends[std::size_t(representative)];
  };
  if (!pooled) {
    emit_single_plane<Index, Int>(local, mesh, corner_of, record_arrangement,
                                  record_cells);
    return true;
  }
  return emit_plane_stack<Index, Int>(world, plane, local, mesh, corner_of,
                                      record_arrangement, record_cells);
}

} // namespace tf::arrangement
