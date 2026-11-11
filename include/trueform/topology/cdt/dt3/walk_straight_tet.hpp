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
#include "../../../exact/meta.hpp"
#include "../../../exact/orient3d.hpp"
#include "./tet_deepest_of.hpp"
#include "./tet_facets.hpp"
#include "./tet_orient3d_sign.hpp"
#include "./tet_position.hpp"
#include "./tet_with_sites.hpp"
#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Walk the segment from a point inside `hint` to the target, the site
/// `target` or a free position, to the finite cell holding it, its facets
/// included, or the hull cell whose base the segment leaves the complex
/// through. The segment starts at the hint's corner zero moved symbolically
/// into the hint, toward its barycenter and then toward its corners one and
/// two by infinitesimals of rising order (orient3d is affine in each point,
/// so the start's side of a plane is the first nonzero of corner zero's, the
/// three other corners' sum, corner one's and corner two's), so it meets no
/// edge and no vertex of the complex before the target: each
/// cell it crosses is left through the one facet the target lies strictly
/// beyond and the segment passes through, a convex cell is crossed once,
/// and the walk ends in any triangulation, Delaunay or not, after the cells
/// the segment crosses. A target on a facet, an edge or a vertex ends in
/// the first cell holding it the segment reaches. A hint that is a hull cell
/// starts from the finite cell across its base.
template <typename Owner, typename Target>
auto walk_straight_tet(const Owner &owner, const Target &target,
                       typename Owner::index_type hint) ->
    typename Owner::index_type {
  using Index = typename Owner::index_type;
  if (owner._corners[std::size_t(hint)][3] == Owner::infinite)
    hint = owner._neighbors[std::size_t(hint)][3];
  const auto start = owner._corners[std::size_t(hint)];
  int target_depth = 0;
  if constexpr (std::is_integral_v<Target>)
    target_depth = tet_deepest_of(owner, Index(target));
  else
    target_depth = target.depth;
  for (auto tet = hint;;) {
    const auto corners = owner._corners[std::size_t(tet)];
    if (corners[3] == Owner::infinite)
      return tet;
    const int depth = std::max(
        target_depth,
        tet_deepest_of(owner, start[0], start[1], start[2], start[3],
                       corners[0], corners[1], corners[2], corners[3]));
    const int exit = tet_with_sites(owner, depth, [&](const auto &sites) {
      using lattice = typename std::decay_t<decltype(sites)>::int_type;
      using T2 = typename tf::exact::meta<lattice>::T2;
      const auto q = [&] {
        if constexpr (std::is_integral_v<Target>)
          return sites[std::size_t(target)].pt;
        else
          return sites.point(target);
      }();
      const auto at = [&sites](Index slot) {
        return sites[std::size_t(slot)].pt;
      };
      const auto value = [&](std::size_t k, Index a, Index b) {
        return tf::exact::orient3d_value_scaled<lattice>(at(start[k]), q,
                                                         at(a), at(b));
      };
      // the side of the plane through the segment's line and the edge a b
      // the moved start lies on
      const auto side = [&](Index a, Index b) {
        T2 v = value(0, a, b);
        if (v == T2(0))
          v = value(1, a, b) + value(2, a, b) + value(3, a, b);
        for (std::size_t k = 1; k < 3 && v == T2(0); ++k)
          v = value(k, a, b);
        return v > T2(0) ? 1 : v < T2(0) ? -1 : 0;
      };
      for (std::size_t slot = 0; slot < 4; ++slot) {
        const auto facet = tet_facet(corners, slot);
        if (tet_orient3d_sign<lattice>(at(facet[0]), at(facet[1]),
                                       at(facet[2]), q) <= 0)
          continue;
        const int ab = side(facet[0], facet[1]);
        const int bc = side(facet[1], facet[2]);
        const int ca = side(facet[2], facet[0]);
        if ((ab >= 0 && bc >= 0 && ca >= 0) || (ab <= 0 && bc <= 0 && ca <= 0))
          return int(slot);
      }
      return -1;
    });
    if (exit < 0)
      return tet;
    tet = owner._neighbors[std::size_t(tet)][std::size_t(exit)];
  }
}

} // namespace tf::topology::cdt::dt3
