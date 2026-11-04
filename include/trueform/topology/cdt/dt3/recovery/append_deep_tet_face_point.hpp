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
#include "../tet_deep_lattice.hpp"
#include "../tet_position.hpp"
#include "./append_tet_face_point.hpp"
#include "./tet_face_lattice_point.hpp"
#include "./tet_face_recovery.hpp"
#include "./tet_point_in_edge_kernel.hpp"
#include "./tet_segment_recovery.hpp"
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Place a point of face `f` below the frame where the face's lattice names
/// none new on it: depth by depth to the deep lattice's cap
/// (@ref tet_deep_lattice), the face's lattice point near where the edge
/// d e crosses its plane (@ref tet_face_lattice_point) that lies in the
/// kernel of the edge's ring (@ref tet_point_in_edge_kernel), placed as
/// @ref append_tet_face_point places one, the shallowest that no site
/// holds and no piece carries. Returns the route of the point placed, or
/// `standing` when no depth to the cap names one.
template <typename DT, typename Index, typename Coord, typename Int,
          typename Facet, typename IsSegment>
auto append_deep_tet_face_point(
    DT &dt, tet_face_recovery<Index, Coord, Int> &recovery,
    const tet_segment_recovery<Index, Coord> &segment, Index f,
    const Facet &facet, const std::pair<int, int> &axes, Index d, Index e,
    Index hint, const IsSegment &is_segment, Index &site)
    -> tet_face_point_route {
  for (int depth = 1; depth <= tet_deep_lattice<Int, Coord>::cap; ++depth) {
    tet_position<Coord> point{};
    if (!tet_face_lattice_point(dt.owner(), facet, axes, d, e, depth, point) ||
        !tet_point_in_edge_kernel(dt.owner(), d, e, point))
      continue;
    const auto route = append_tet_face_point(dt, recovery, segment, f, point,
                                             hint, is_segment, site);
    if (route != tet_face_point_route::standing &&
        route != tet_face_point_route::on_piece)
      return route;
  }
  return tet_face_point_route::standing;
}

} // namespace tf::topology::cdt::dt3
