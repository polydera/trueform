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
#include "../../../../core/range.hpp"
#include "../../../../exact/meta.hpp"
#include "../tet_deepest_of.hpp"
#include "../tet_minted_site.hpp"
#include "../tet_position.hpp"
#include "../tet_site_position.hpp"
#include "../tet_with_sites.hpp"
#include "./tet_face_recovery.hpp"
#include "./tet_segment_recovery.hpp"
#include "./walk_to_tet_point.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// What placing a face point did: a fresh site stands at it; a site
/// already stood there; it lies on a piece, which it may not split; the
/// site budget refused it; the append refused, which publishes no cells.
enum class tet_face_point_route : std::uint8_t {
  placed,
  standing,
  on_piece,
  budget,
  failed
};

/// Place the point at the position `pt` of face `f` into the complex `dt`:
/// the cell it lies in is walked to from `hint`; a site standing at it is named
/// in `site`; a point on an edge of that cell that is a segment, read at the
/// deepest depth of the three, is refused; otherwise the point is appended
/// as a one-site batch — the stellar split in this world — and named in
/// `site` and in the face points. `is_segment` answers for the pieces.
template <typename DT, typename Index, typename Coord, typename Int,
          typename IsSegment>
auto append_tet_face_point(DT &dt,
                           tet_face_recovery<Index, Coord, Int> &recovery,
                           const tet_segment_recovery<Index, Coord> &segment,
                           Index f, const tet_position<Coord> &pt, Index hint,
                           const IsSegment &is_segment, Index &site)
    -> tet_face_point_route {
  site = DT::k_none;
  const Index cell = walk_to_tet_point(dt, pt, hint);
  if (cell == DT::k_none)
    return tet_face_point_route::failed;
  const auto &owner = dt.owner();
  const auto corners = dt.tets()[std::size_t(cell)];
  for (auto corner : corners)
    if (tet_site_position(owner, corner) == pt) {
      site = corner;
      return tet_face_point_route::standing;
    }
  const auto on_edge = [&owner, &pt](Index x, Index y) {
    const int depth = std::max(tet_deepest_of(owner, x, y), pt.depth);
    return tet_with_sites(owner, depth, [&](const auto &sites) {
      using lattice = typename std::decay_t<decltype(sites)>::int_type;
      using T1 = typename tf::exact::meta<lattice>::T1;
      using T2 = typename tf::exact::meta<lattice>::T2;
      const auto &px = sites[std::size_t(x)].pt;
      const auto &py = sites[std::size_t(y)].pt;
      const auto p = sites.point(pt);
      T1 u[3], v[3], w[3];
      for (std::size_t k = 0; k < 3; ++k) {
        u[k] = T1(py[k]) - T1(px[k]);
        v[k] = T1(p[k]) - T1(px[k]);
        w[k] = T1(p[k]) - T1(py[k]);
      }
      for (std::size_t k = 0; k < 3; ++k) {
        const std::size_t i = (k + 1) % 3, j = (k + 2) % 3;
        if (T2(u[i]) * v[j] - T2(u[j]) * v[i] != 0)
          return false;
      }
      T2 along(0), back(0);
      for (std::size_t k = 0; k < 3; ++k) {
        along += T2(v[k]) * u[k];
        back -= T2(w[k]) * u[k];
      }
      return along > 0 && back > 0;
    });
  };
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = i + 1; j < 4; ++j)
      if (on_edge(corners[i], corners[j]) && is_segment(corners[i], corners[j]))
        return tet_face_point_route::on_piece;
  if (dt.n_sites() + 1 > segment.site_budget)
    return tet_face_point_route::budget;
  const tet_minted_site<Index, Coord> minted{Index(dt.next_name()), pt};
  const std::size_t base = dt.index_map().f().size();
  if (!dt.append_sites(tf::make_range(&minted, std::size_t(1))))
    return tet_face_point_route::failed;
  const Index slot = dt.index_map().f()[base];
  if (slot == DT::k_none)
    return tet_face_point_route::failed;
  site = slot;
  recovery.points.push_back({f, slot});
  return tet_face_point_route::placed;
}

} // namespace tf::topology::cdt::dt3
