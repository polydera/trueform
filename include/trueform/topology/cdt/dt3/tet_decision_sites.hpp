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
#include "../../../exact/vertex.hpp"
#include "./tet_aligned_position.hpp"
#include "./tet_position.hpp"
#include "./tet_site_position.hpp"
#include "./tetrahedralization_owner.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The sites one decision reads, every one at the decision's depth, so the
/// values it compares share one scale: indexed by slot, a site's name and
/// point; `point(p)` a free position read beside them, `depth` the depth
/// they are read at and `int_type` the lattice their values run on. At depth
/// zero each point is the site's floor on the `Int` lattice, read shallow.
template <typename Index, typename Coord, typename Int>
struct tet_shallow_decision {
  using index_type = Index;
  using int_type = Int;
  using coord_type = Coord;
  static constexpr bool deep = false;
  static constexpr int depth = 0;
  tet_site_lanes<Index, Coord> lanes;

  auto operator[](std::size_t slot) const
      -> const tf::exact::vertex<Index, Coord> & {
    return lanes.shallow(Index(slot));
  }
  auto point(const tet_position<Coord> &p) const -> tf::exact::pt3<Coord> {
    return tet_shallow_position(p);
  }
};

/// The sites of `owner` read on the frame, for a decision every corner of
/// which stands on it.
template <typename Index, typename Coord, typename Int>
auto tet_shallow_decision_of(
    const tetrahedralization_owner<Index, Coord, Int> &owner)
    -> tet_shallow_decision<Index, Coord, Int> {
  return {tet_site_lanes_of(owner)};
}

/// Below the frame each point is the site's position aligned `depth` bits
/// down in the deep lattice's `Wide` coordinate, on its `Int` lattice
/// (@ref tet_deep_lattice).
template <typename Index, typename Coord, typename Int, typename Wide>
struct tet_aligned_decision {
  using index_type = Index;
  using int_type = Int;
  using coord_type = Wide;
  static constexpr bool deep = true;
  tet_site_lanes<Index, Coord> lanes;
  int depth;

  auto operator[](std::size_t slot) const -> tf::exact::vertex<Index, Wide> {
    return {lanes.name(Index(slot)),
            tet_aligned_position<Wide>(lanes.position(Index(slot)), depth)};
  }
  auto point(const tet_position<Coord> &p) const -> tf::exact::pt3<Wide> {
    return tet_aligned_position<Wide>(p, depth);
  }
};

} // namespace tf::topology::cdt::dt3
