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
#include "../../../core/config/verify.hpp"
#include "../../../core/range.hpp"
#include "../../../exact/vertex.hpp"
#include "./tet_deep_site.hpp"
#include "./tet_position.hpp"
#include "./tetrahedralization_owner.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The one reader of the sites' positions, a value over the owner's lanes:
/// per slot its name and floor, the deep rows and their tickets by name, and
/// the deepest depth they state (@ref tetrahedralization_owner). Every read
/// is a ticket: no table is searched. A shallow read verifies that
/// its site stands on the frame, so a floor is never taken for the position
/// of a site below it.
template <typename Index, typename Coord> struct tet_site_lanes {
  tf::range<const tf::exact::vertex<Index, Coord> *, tf::dynamic_size> floors;
  tf::range<const tet_deep_site<Index> *, tf::dynamic_size> rows;
  tf::range<const Index *, tf::dynamic_size> at;
  int deepest;

  auto name(Index slot) const -> Index { return floors[std::size_t(slot)].id; }
  auto row(Index slot) const -> Index {
    const Index key = name(slot);
    return std::size_t(key) < at.size() ? at[std::size_t(key)] : Index(-1);
  }
  auto depth(Index slot) const -> int {
    if (!deepest)
      return 0;
    const Index ticket = row(slot);
    return ticket < 0 ? 0 : rows[std::size_t(ticket)].depth;
  }
  auto position(Index slot) const -> tet_position<Coord> {
    tet_position<Coord> p;
    p.floor = floors[std::size_t(slot)].pt;
    if (!deepest)
      return p;
    const Index row = this->row(slot);
    if (row >= 0) {
      p.depth = rows[std::size_t(row)].depth;
      p.residual = rows[std::size_t(row)].residual;
    }
    return p;
  }
  auto shallow(Index slot) const -> const tf::exact::vertex<Index, Coord> & {
    if constexpr (tf::core::verification)
      tf::core::verify(!depth(slot),
                       "a shallow read names a site on the frame");
    return floors[std::size_t(slot)];
  }
};

template <typename Index, typename Coord, typename Int>
auto tet_site_lanes_of(const tetrahedralization_owner<Index, Coord, Int> &owner)
    -> tet_site_lanes<Index, Coord> {
  return {tf::make_range(owner._floors), tf::make_range(owner._deep),
          tf::make_range(owner._deep_at), owner._deepest};
}

template <typename Index, typename Coord, typename Int>
auto tet_site_count(const tetrahedralization_owner<Index, Coord, Int> &owner)
    -> std::size_t {
  return tet_site_lanes_of(owner).floors.size();
}

template <typename Index, typename Coord, typename Int>
auto tet_site_name(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type slot)
    -> Index {
  return tet_site_lanes_of(owner).name(slot);
}

/// How deep below the frame site `slot` stands.
template <typename Index, typename Coord, typename Int>
auto tet_site_depth(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type slot)
    -> int {
  return tet_site_lanes_of(owner).depth(slot);
}

/// The depth no site has stood below, the deepest the rows state: zero
/// while every site has stood on the frame. A site withdrawn below the
/// frame leaves it standing.
template <typename Index, typename Coord, typename Int>
auto tet_deepest_depth(const tetrahedralization_owner<Index, Coord, Int> &owner)
    -> int {
  return tet_site_lanes_of(owner).deepest;
}

/// The position site `slot` stands on.
template <typename Index, typename Coord, typename Int>
auto tet_site_position(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type slot)
    -> tet_position<Coord> {
  return tet_site_lanes_of(owner).position(slot);
}

/// Site `slot`'s position read shallow, in place.
template <typename Index, typename Coord, typename Int>
auto tet_shallow_point(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type slot)
    -> const tf::exact::pt3<Coord> & {
  return tet_site_lanes_of(owner).shallow(slot).pt;
}

/// Site `slot` as Simulation of Simplicity ranks it: its name and its
/// position read shallow, in place.
template <typename Index, typename Coord, typename Int>
auto tet_shallow_site(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type slot)
    -> const tf::exact::vertex<Index, Coord> & {
  return tet_site_lanes_of(owner).shallow(slot);
}

/// Every site as such a record, in slot order: the sites a complex
/// publishes, every one standing on the frame.
template <typename Index, typename Coord, typename Int>
auto tet_shallow_sites_of(
    const tetrahedralization_owner<Index, Coord, Int> &owner)
    -> tf::range<const tf::exact::vertex<Index, Coord> *, tf::dynamic_size> {
  const auto lanes = tet_site_lanes_of(owner);
  if constexpr (tf::core::verification)
    for (std::size_t slot = 0; slot < lanes.floors.size(); ++slot)
      tf::core::verify(!lanes.depth(Index(slot)),
                       "the published sites stand on the frame");
  return lanes.floors;
}

} // namespace tf::topology::cdt::dt3
