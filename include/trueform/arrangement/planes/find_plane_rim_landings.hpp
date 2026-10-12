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

/// THE RIM LAW: a face's cut never leaves the face. The pierce a cut ends at
/// is inside its face exactly, and the one rounding may still carry it across
/// a side the face stands on; the cut then bounds a sliver the face does not
/// cover on either side, nothing crosses anything, and the cut's pages lose
/// the face. On the lattice that point IS on the side, and the cascade
/// demotes it there: it lands on the wall between the sliver and the face's
/// own region, which splits that wall at an identity that already stands.
///
/// A cut that leaves the face past a corner bounds a sliver both of whose
/// other sides are the face's walls: there it is the face's corner that
/// stands on the cut, and the corner lands on it. Either way the sliver's
/// sides become one chain, and only a point strictly inside the span of the
/// constraint it lands on is stated.
///
/// `local.cons_cut` holds `{constraint, member}` per member stating a cut,
/// ascending; `covers(member, triangle)` and `walls(member, constraint)` are
/// the carrier's coverage and boundary.
template <typename Index, typename Int, typename Mesh, typename Local,
          typename Covers, typename Walls>
auto find_plane_rim_landings(
    const Mesh &mesh, const Local &local, const Covers &covers,
    const Walls &walls,
    tf::buffer<plane_rim_landing<Index, typename tf::exact::meta<Int>::param_type>>
        &landings) -> void {
  using param_t = typename tf::exact::meta<Int>::param_type;
  landings.clear();
  if (local.cons_cut.size() == 0)
    return;
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
  mesh.for_each_face_adjacency([&](Index triangle, Index v0, Index v1,
                                   Index v2, Index,
                                   const std::array<Index, 3> &neighbors,
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
        int bounding_walls = 0;
        for (int side = 0; side < 3; ++side) {
          if (side == edge ||
              owners[std::size_t(side)].input_id == Index(-1))
            continue;
          const auto wall = root_of(owners[std::size_t(side)].input_id);
          if (!walls(member, wall))
            continue;
          ++bounding_walls;
          const auto inside = neighbors[std::size_t(side)];
          if (inside != Index(-1) && covers(member, inside))
            landed = land(wall, corners[std::size_t((side + 2) % 3)]) || landed;
        }
        if (!landed && bounding_walls == 2)
          land(cut, corners[std::size_t((edge + 2) % 3)]);
      }
    }
  });
  if (landings.size() < 2)
    return;
  const auto key = [](const plane_rim_landing<Index, param_t> &x) {
    return std::make_tuple(x.constraint, x.point);
  };
  std::sort(landings.begin(), landings.end(),
            [&key](const auto &x, const auto &y) { return key(x) < key(y); });
  landings.erase_till_end(std::unique(
      landings.begin(), landings.end(),
      [&key](const auto &x, const auto &y) { return key(x) == key(y); }));
}

} // namespace tf::arrangement
