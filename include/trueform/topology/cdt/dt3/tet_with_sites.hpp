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
#include "./tet_decision_sites.hpp"
#include "./tet_deep_lattice.hpp"
#include "./tet_site_position.hpp"
#include "./tetrahedralization_owner.hpp"

namespace tf::topology::cdt::dt3 {

/// Decide `f` on the sites `lanes` reads aligned `depth` bits below the
/// frame on the deep lattice of the `Int` lattice stored in `Coord`.
template <typename Int, typename Index, typename Coord, typename F>
auto tet_with_aligned_sites(const tet_site_lanes<Index, Coord> &lanes,
                            int depth, const F &f) {
  using lattice = tet_deep_lattice<Int, Coord>;
  return f(tet_aligned_decision<Index, Coord, typename lattice::int_type,
                                typename lattice::coord_type>{lanes, depth});
}

/// Decide `f` on the sites of `owner` read `depth` bits below the frame
/// (@ref tet_decision_sites): on the frame at zero, aligned on the owner's
/// deep lattice otherwise. A decision whose values are compared with each
/// other reads them at the deepest depth among every corner they read.
template <typename Index, typename Coord, typename Int, typename F>
auto tet_with_sites(const tetrahedralization_owner<Index, Coord, Int> &owner,
                    int depth, const F &f)
    -> decltype(f(tet_shallow_decision<Index, Coord, Int>{
        tet_site_lanes_of(owner)})) {
  const auto lanes = tet_site_lanes_of(owner);
  if constexpr (tet_deep_lattice<Int, Coord>::cap > 0)
    if (depth)
      return tet_with_aligned_sites<Int>(lanes, depth, f);
  return f(tet_shallow_decision<Index, Coord, Int>{lanes});
}

} // namespace tf::topology::cdt::dt3
