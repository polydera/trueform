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
#include "./tet_deep_lattice.hpp"
#include "./tet_deepest_of.hpp"
#include "./tet_orient3d_sign.hpp"
#include "./tet_position.hpp"
#include "./tet_site_position.hpp"
#include "./tet_with_sites.hpp"
#include "./tetrahedralization_owner.hpp"
#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// The orient3d sign of the sites `a`, `b`, `c`, `d` read at the deepest
/// depth among them (@ref tet_with_sites).
template <typename Index, typename Coord, typename Int>
auto tet_orient3d_deep_sites(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type a,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type b,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type d) -> int {
  return tet_with_sites(
      owner, tet_deepest_of(owner, a, b, c, d), [&](const auto &sites) {
        using lattice = typename std::decay_t<decltype(sites)>::int_type;
        return tet_orient3d_sign<lattice>(
            sites[std::size_t(a)].pt, sites[std::size_t(b)].pt,
            sites[std::size_t(c)].pt, sites[std::size_t(d)].pt);
      });
}

/// The orient3d sign of the sites `a`, `b`, `c`, `d`: on their floors while
/// every site stands on the frame, the per-model flag the one check the
/// frame's path pays, else read at the deepest depth among them.
template <typename Index, typename Coord, typename Int>
auto tet_orient3d_sites(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type a,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type b,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type d) -> int {
  const auto lanes = tet_site_lanes_of(owner);
  if constexpr (tet_deep_lattice<Int, Coord>::cap > 0)
    if (lanes.deepest)
      return tet_orient3d_deep_sites(owner, a, b, c, d);
  return tet_orient3d_sign<Int>(lanes.shallow(a).pt, lanes.shallow(b).pt,
                                lanes.shallow(c).pt, lanes.shallow(d).pt);
}

/// The same sign in the reference's reading (@ref tet_orient3d_below):
/// positive when `d` lies below the plane of `a`, `b`, `c`.
template <typename Index, typename Coord, typename Int>
auto tet_orient3d_below_sites(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type a,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type b,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type d) -> int {
  return -tet_orient3d_sites(owner, a, b, c, d);
}

/// The orient3d sign of the sites `a`, `b`, `c` and the position `p`, read
/// at the deepest depth among them.
template <typename Index, typename Coord, typename Int>
auto tet_orient3d_toward(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type a,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type b,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c,
    const tet_position<Coord> &p) -> int {
  return tet_with_sites(
      owner, std::max(tet_deepest_of(owner, a, b, c), p.depth),
      [&](const auto &sites) {
        using lattice = typename std::decay_t<decltype(sites)>::int_type;
        return tet_orient3d_sign<lattice>(
            sites[std::size_t(a)].pt, sites[std::size_t(b)].pt,
            sites[std::size_t(c)].pt, sites.point(p));
      });
}

} // namespace tf::topology::cdt::dt3
