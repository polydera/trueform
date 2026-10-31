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
#include "../../exact/meta.hpp"
#include "./plane_landing_parameter.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>

namespace tf::arrangement {

/// A point the rim law lands on a constraint it stands on: `constraint` and
/// `point` are prepared indices, `parameter` the dyadic position along the
/// constraint in the order the plane fed it.
template <typename Index, typename Param> struct plane_rim_landing {
  Index constraint;
  Index point;
  Param parameter;
};

/// The rim law's scratch, reused across a block's planes: the uncovered cut
/// occurrences no wall landing reached, as `{triangle, edge, entry}` with
/// `entry` the cut's position in `cons_cut`; the carrier's corners and
/// neighbours, read only when one is; and the walk's frontier, as
/// `{triangle, corner}` with `corner` the one off the edge it was entered by.
template <typename Index> struct plane_rim_walk {
  tf::buffer<std::array<Index, 3>> unlanded;
  tf::buffer<std::array<Index, 3>> corners;
  tf::buffer<std::array<Index, 3>> neighbors;
  tf::buffer<std::array<Index, 2>> frontier;
  tf::buffer<Index> visited;
};

/// THE RIM LAW: a face's cut never leaves the face. The pierce a cut ends at
/// is inside its face exactly, and the one rounding may still carry it across
/// a side the face stands on; the cut then bounds a sliver the face does not
/// cover on either side, nothing crosses anything, and the cut's pages lose
/// the face. On the lattice that point IS on the side, and the cascade
/// demotes it there: it lands on the wall between the sliver and the face's
/// own region, which splits that wall at an identity that already stands.
///
/// Where no such wall lands anything, the sliver lies between the cut and the
/// face's own corners — a cut that leaves the face past a corner, or a side
/// whose points were rounded one by one across it. Every corner of the sliver
/// within one lattice unit of the cut, strictly inside its span, stands on
/// the cut, and all of them land on it at once: the sliver is walked from the
/// cut across the triangulation's free edges, entering a triangle only where
/// the corner off its entry edge stands on the cut. Either way the sliver's
/// sides become one chain, and only a point strictly inside the span of the
/// constraint it lands on is stated.
///
/// `local.cons_cut` holds `{constraint, member}` per member stating a cut,
/// ascending; `covers(member, triangle)` and `walls(member, constraint)` are
/// the carrier's coverage and boundary. Returns the uncovered cuts the law
/// could not land when it landed nothing — a sliver wider than the lattice's
/// own rounding — which the plane then emits in no slot.
template <typename Index, typename Int, typename Mesh, typename Local,
          typename Covers, typename Walls>
auto find_plane_rim_landings(
    const Mesh &mesh, const Local &local, const Covers &covers,
    const Walls &walls, plane_rim_walk<Index> &walk,
    tf::buffer<
        plane_rim_landing<Index, typename tf::exact::meta<Int>::param_type>>
        &landings) -> std::size_t {
  using param_t = typename tf::exact::meta<Int>::param_type;
  landings.clear();
  if (local.cons_cut.size() == 0)
    return 0;
  walk.unlanded.clear();
  const auto points = mesh.points();
  const auto root_of = [&local](Index input) {
    return local.cons_root.size() == 0 ? input
                                       : local.cons_root[std::size_t(input)];
  };
  // the point at `vertex`, landed on `constraint` where its foot falls
  // strictly inside the constraint's span
  const auto land = [&](Index constraint, Index vertex) {
    const auto point = local.rep[std::size_t(vertex)];
    const auto a = local.cons[std::size_t(constraint) * 2];
    const auto b = local.cons[std::size_t(constraint) * 2 + 1];
    if (point == Index(-1) || point == a || point == b)
      return false;
    const auto parameter = plane_landing_parameter<Int>(
        local.pts2[std::size_t(a)], local.pts2[std::size_t(b)],
        points[std::size_t(vertex)]);
    if (parameter == param_t(-1))
      return false;
    landings.push_back({constraint, point, parameter});
    return true;
  };
  mesh.for_each_face_adjacency([&](Index triangle, Index v0, Index v1, Index v2,
                                   Index, const std::array<Index, 3> &neighbors,
                                   const std::array<bool, 3> &) {
    const auto owners = mesh.face_constraint_owners(triangle);
    const std::array<Index, 3> corners{v0, v1, v2};
    for (int edge = 0; edge < 3; ++edge) {
      if (owners[std::size_t(edge)].input_id == Index(-1))
        continue;
      const auto cut = root_of(owners[std::size_t(edge)].input_id);
      auto at = std::lower_bound(local.cons_cut.begin(), local.cons_cut.end(),
                                 std::array<Index, 2>{cut, Index(-1)});
      for (; at != local.cons_cut.end() && (*at)[0] == cut; ++at) {
        const auto member = (*at)[1];
        const auto across = neighbors[std::size_t(edge)];
        if (covers(member, triangle) ||
            (across != Index(-1) && covers(member, across)))
          continue;
        bool landed = false;
        for (int side = 0; side < 3; ++side) {
          if (side == edge || owners[std::size_t(side)].input_id == Index(-1))
            continue;
          const auto wall = root_of(owners[std::size_t(side)].input_id);
          if (!walls(member, wall))
            continue;
          const auto inside = neighbors[std::size_t(side)];
          if (inside != Index(-1) && covers(member, inside))
            landed = land(wall, corners[std::size_t((side + 2) % 3)]) || landed;
        }
        if (!landed)
          walk.unlanded.push_back(
              {triangle, Index(edge), Index(at - local.cons_cut.begin())});
      }
    }
  });
  if (walk.unlanded.size() != 0) {
    const auto n_triangles = std::size_t(mesh.region_labels().size());
    walk.corners.allocate(n_triangles);
    walk.neighbors.allocate(n_triangles);
    mesh.for_each_face_adjacency([&walk](Index triangle, Index v0, Index v1,
                                         Index v2, Index,
                                         const std::array<Index, 3> &neighbors,
                                         const std::array<bool, 3> &) {
      walk.corners[std::size_t(triangle)] = {v0, v1, v2};
      walk.neighbors[std::size_t(triangle)] = neighbors;
    });
    for (const auto &occurrence : walk.unlanded) {
      const auto cut = local.cons_cut[std::size_t(occurrence[2])][0];
      const auto a = local.cons[std::size_t(cut) * 2];
      const auto b = local.cons[std::size_t(cut) * 2 + 1];
      walk.frontier.clear();
      walk.visited.clear();
      walk.frontier.push_back({occurrence[0], (occurrence[1] + Index(2)) % 3});
      while (walk.frontier.size() != 0) {
        const auto [triangle, off] = walk.frontier[walk.frontier.size() - 1];
        walk.frontier.pop_back();
        if (std::find(walk.visited.begin(), walk.visited.end(), triangle) !=
            walk.visited.end())
          continue;
        walk.visited.push_back(triangle);
        const auto &corners = walk.corners[std::size_t(triangle)];
        const auto vertex = corners[std::size_t(off)];
        const auto parameter = plane_unit_landing_parameter<Int>(
            local.pts2[std::size_t(a)], local.pts2[std::size_t(b)],
            points[std::size_t(vertex)]);
        if (parameter == param_t(-1))
          continue;
        const auto point = local.rep[std::size_t(vertex)];
        if (point != Index(-1) && point != a && point != b)
          landings.push_back({cut, point, parameter});
        const auto owners = mesh.face_constraint_owners(triangle);
        for (Index edge = 0; edge < 3; ++edge) {
          const auto next =
              walk.neighbors[std::size_t(triangle)][std::size_t(edge)];
          if (edge == (off + Index(1)) % 3 || next == Index(-1) ||
              owners[std::size_t(edge)].input_id != Index(-1))
            continue;
          const auto u = corners[std::size_t(edge)];
          const auto v = corners[std::size_t((edge + Index(1)) % 3)];
          const auto &beyond = walk.corners[std::size_t(next)];
          Index corner = 0;
          while (beyond[std::size_t(corner)] == u ||
                 beyond[std::size_t(corner)] == v)
            ++corner;
          walk.frontier.push_back({next, corner});
        }
      }
    }
  }
  if (landings.size() == 0) {
    std::sort(walk.unlanded.begin(), walk.unlanded.end(),
              [](const auto &x, const auto &y) { return x[2] < y[2]; });
    std::size_t uncovered = 0;
    for (std::size_t i = 0; i < walk.unlanded.size(); ++i)
      uncovered +=
          std::size_t(i == 0 || walk.unlanded[i][2] != walk.unlanded[i - 1][2]);
    return uncovered;
  }
  const auto key = [](const plane_rim_landing<Index, param_t> &x) {
    return std::make_tuple(x.constraint, x.point);
  };
  std::sort(landings.begin(), landings.end(),
            [&key](const auto &x, const auto &y) { return key(x) < key(y); });
  landings.erase_till_end(std::unique(
      landings.begin(), landings.end(),
      [&key](const auto &x, const auto &y) { return key(x) == key(y); }));
  return 0;
}

} // namespace tf::arrangement
