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
#include "../../../exact/insphere.hpp"
#include "./tet_deep_lattice.hpp"
#include "./tet_deepest_of.hpp"
#include "./tet_site_position.hpp"
#include "./tet_with_sites.hpp"
#include "./tetrahedralization_owner.hpp"
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Whether the site `e` conflicts with the circumsphere of the positively
/// oriented cell `c0 c1 c2 c3`, symbolic ranks breaking cospherical ties,
/// every site read at the deepest depth among the five
/// (@ref tet_with_sites).
template <typename Index, typename Coord, typename Int>
auto tet_insphere_conflict_deep_sites(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c0,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c1,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c2,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c3,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type e)
    -> bool {
  return tet_with_sites(
      owner, tet_deepest_of(owner, c0, c1, c2, c3, e), [&](const auto &sites) {
        using lattice = typename std::decay_t<decltype(sites)>::int_type;
        return tf::exact::insphere_conflict_scaled<lattice>(
            sites[std::size_t(c0)], sites[std::size_t(c1)],
            sites[std::size_t(c2)], sites[std::size_t(c3)],
            sites[std::size_t(e)], 1);
      });
}

/// Whether the site `e` conflicts with the circumsphere of the positively
/// oriented cell `c0 c1 c2 c3`: on their floors while every site stands on
/// the frame, the per-model flag the one check the frame's path pays, else
/// read at the deepest depth among the five.
template <typename Index, typename Coord, typename Int>
auto tet_insphere_conflict_sites(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c0,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c1,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c2,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type c3,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type e)
    -> bool {
  const auto lanes = tet_site_lanes_of(owner);
  if constexpr (tet_deep_lattice<Int, Coord>::cap > 0)
    if (lanes.deepest)
      return tet_insphere_conflict_deep_sites(owner, c0, c1, c2, c3, e);
  return tf::exact::insphere_conflict_scaled<Int>(
      lanes.shallow(c0), lanes.shallow(c1), lanes.shallow(c2),
      lanes.shallow(c3), lanes.shallow(e), 1);
}

} // namespace tf::topology::cdt::dt3
