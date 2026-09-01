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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/vertex.hpp"
#include "../../topology/topo_id.hpp"
#include "./plane_arrangement_arena.hpp"
#include "./plane_arrangement_census.hpp"
#include "./plane_refinement_plan.hpp"
#include "./plane_triangulation_types.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace tf::arrangement {

/// CORE. Materialize once, at the ownership boundary: the planes in order, so
/// every range is a prefix of the walk.
///
/// A round aggregates its task blocks in input order, so a build that visited
/// every plane once and rebuilt none already holds a plane-major arena: every
/// shift is zero and the walk below would copy the store onto itself. THE
/// ARENA IS THE PRODUCT then, and it moves.
template <typename Index, typename Int, typename World>
auto materialize_plane_products(
    const World &world, plane_arrangement_arena<Index, Int> &arena,
    bool record_arrangement, bool record_cells, Index base_created, Index n_flat,
    tf::buffer<Index> &plane_offsets,
    tf::buffer<std::array<Index, 3>> &triangles,
    tf::buffer<std::array<Index, 3>> &slot_parents,
    tf::buffer<std::array<tf::topo_id<short>, 3>> &corner_subs,
    tf::buffer<Index> &coplanar_of,
    tf::buffer<tf::arrangement::coplanar_descriptor<Index>> &coplanar,
    tf::buffer<char> &stacked, tf::buffer<Index> &triangle_cells,
    tf::buffer<std::array<Index, 2>> &face_range,
    tf::buffer<tf::exact::pt3<Int>> &direct_created_points,
    tf::buffer<Index> &created_class, plane_arrangement_census &census,
    plane_refinement_census &refinement_census) -> void {
  plane_offsets.allocate(std::size_t(world.n_planes()) + 1);
  plane_offsets[0] = 0;
  bool arena_is_product = true;
  for (Index p = 0; p < world.n_planes(); ++p) {
    const auto range = arena.range[std::size_t(p)];
    arena_is_product =
        arena_is_product && range[0] == plane_offsets[std::size_t(p)];
    plane_offsets[std::size_t(p) + 1] =
        plane_offsets[std::size_t(p)] + (range[1] - range[0]);
  }
  const auto n = std::size_t(plane_offsets[std::size_t(world.n_planes())]);
  arena_is_product = arena_is_product && arena.tris.size() == n;
  tf::buffer<Index> cop_base;
  cop_base.allocate(std::size_t(world.n_planes()) + 1);
  cop_base[0] = 0;
  for (Index p = 0; p < world.n_planes(); ++p) {
    const auto cop = arena.cop_range[std::size_t(p)];
    arena_is_product =
        arena_is_product && cop[0] == cop_base[std::size_t(p)];
    cop_base[std::size_t(p) + 1] = cop_base[std::size_t(p)] + (cop[1] - cop[0]);
  }
  // THE INTERIOR IDENTITIES ARE MINTED HERE, over what the arena finally
  // holds: a plane a later round rebuilt replaced its own points, so nothing
  // is named that no triangle stands on. Plane order makes it deterministic.
  tf::buffer<Index> stn_base;
  stn_base.allocate(std::size_t(world.n_planes()) + 1);
  stn_base[0] = 0;
  for (Index p = 0; p < world.n_planes(); ++p) {
    const auto stn = arena.stn_range[std::size_t(p)];
    stn_base[std::size_t(p) + 1] = stn_base[std::size_t(p)] + (stn[1] - stn[0]);
  }
  const auto n_steiners = std::size_t(stn_base[std::size_t(world.n_planes())]);
  const auto direct_base = direct_created_points.size();
  const auto class_base = created_class.size();
  const auto created_base = base_created + Index(class_base);
  direct_created_points.reallocate(direct_base + n_steiners);
  created_class.reallocate(class_base + n_steiners);
  if (arena_is_product && n_steiners == 0) {
    triangles = std::move(arena.tris);
    corner_subs = std::move(arena.subs);
    if (record_arrangement) {
      slot_parents = std::move(arena.slots);
      coplanar_of = std::move(arena.coplanar_of);
      coplanar = std::move(arena.coplanar);
      stacked = std::move(arena.stacked);
    }
    if (record_cells)
      triangle_cells = std::move(arena.cell_of);
    face_range = std::move(arena.face_range);
    census.triangles = n;
    census.created = created_class.size();
    refinement_census.steiners = 0;
    return;
  }
  triangles.allocate(n);
  corner_subs.allocate(n);
  if (record_arrangement) {
    slot_parents.allocate(n);
    coplanar_of.allocate(n);
    stacked.allocate(n);
    coplanar.allocate(std::size_t(cop_base[std::size_t(world.n_planes())]));
  }
  if (record_cells)
    triangle_cells.allocate(n);
  face_range.allocate(arena.face_range.size());
  tf::parallel_for_each(
      tf::make_sequence_range(world.n_planes()),
      [&](Index p) {
        const auto range = arena.range[std::size_t(p)];
        const auto shift = plane_offsets[std::size_t(p)] - range[0];
        const auto cop = arena.cop_range[std::size_t(p)];
        const auto cop_shift = cop_base[std::size_t(p)] - cop[0];
        const auto stn = arena.stn_range[std::size_t(p)];
        const auto stn_shift = stn_base[std::size_t(p)];
        for (auto s = stn[0]; s < stn[1]; ++s) {
          const auto direct =
              direct_base + std::size_t(s - stn[0] + stn_shift);
          direct_created_points[direct] = arena.steiners[std::size_t(s)];
          created_class[class_base + std::size_t(s - stn[0] + stn_shift)] =
              -Index(direct) - Index(1);
        }
        const auto steiner_flat = n_flat + created_base + stn_shift;
        for (auto t = range[0]; t < range[1]; ++t) {
          const auto at = std::size_t(t + shift);
          auto triangle = arena.tris[std::size_t(t)];
          for (int corner = 0; corner < 3; ++corner)
            if (triangle[std::size_t(corner)] < Index(0))
              triangle[std::size_t(corner)] =
                  steiner_flat + (Index(-2) - triangle[std::size_t(corner)]);
          triangles[at] = triangle;
          corner_subs[at] = arena.subs[std::size_t(t)];
          if (record_arrangement) {
            slot_parents[at] = arena.slots[std::size_t(t)];
            stacked[at] = arena.stacked[std::size_t(t)];
            const auto link = arena.coplanar_of[std::size_t(t)];
            coplanar_of[at] = link == Index(-1) ? Index(-1) : link + cop_shift;
          }
        }
        if (record_cells)
          for (auto t = range[0]; t < range[1]; ++t)
            triangle_cells[std::size_t(t + shift)] =
                arena.cell_of[std::size_t(t)];
        if (record_arrangement)
          for (auto d = cop[0]; d < cop[1]; ++d) {
            auto descriptor = arena.coplanar[std::size_t(d)];
            descriptor.survivor += shift;
            coplanar[std::size_t(d + cop_shift)] = descriptor;
          }
        const auto n_members = std::size_t(world.member_count(p));
        for (std::size_t m = 0; m < n_members; ++m) {
          const auto face = world.member(p, Index(m));
          const auto own = arena.face_range[std::size_t(face)];
          face_range[std::size_t(face)] = {own[0] + shift, own[1] + shift};
        }
      },
      tf::checked);
  census.triangles = n;
  census.created = created_class.size();
  refinement_census.steiners = n_steiners;
}

} // namespace tf::arrangement
