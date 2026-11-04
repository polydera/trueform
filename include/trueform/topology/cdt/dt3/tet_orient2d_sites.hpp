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
#include "./recovery/tet_orient2d_sign.hpp"
#include "./recovery/tet_projected_point.hpp"
#include "./tet_deepest_of.hpp"
#include "./tet_with_sites.hpp"
#include "./tetrahedralization_owner.hpp"
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// The orient2d sign of the sites `a`, `b`, `c` read on the projection
/// `axes`, at the deepest depth among them.
template <typename Index, typename Coord, typename Int>
auto tet_orient2d_sites(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    const std::pair<int, int> &axes,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type a,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type b,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c) -> int {
  return tet_with_sites(
      owner, tet_deepest_of(owner, a, b, c), [&](const auto &sites) {
        using lattice = typename std::decay_t<decltype(sites)>::int_type;
        return tet_orient2d_sign<lattice>(
            tet_projected_point(sites[std::size_t(a)].pt, axes),
            tet_projected_point(sites[std::size_t(b)].pt, axes),
            tet_projected_point(sites[std::size_t(c)].pt, axes));
      });
}

} // namespace tf::topology::cdt::dt3
