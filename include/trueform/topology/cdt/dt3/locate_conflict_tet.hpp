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
#include "./step_tet_walk.hpp"
#include "./tet_protected_count.hpp"
#include "./walk_straight_tet.hpp"

namespace tf::topology::cdt::dt3 {

/// Walk from `hint` to a finite cell holding the site or a hull cell seeing
/// it. On a Delaunay complex the walk steps through any facet the site lies
/// beyond and ends; once facets are protected, or a flip has run
/// (`_delaunay`), the complex is not, such a walk can cycle, and the walk
/// follows the segment from inside the hint to the site instead
/// (@ref walk_straight_tet), which ends in any triangulation.
template <typename Owner>
auto locate_conflict_tet(const Owner &owner, typename Owner::index_type site,
                         typename Owner::index_type hint) ->
    typename Owner::index_type {
  if (!(tet_protected_count(owner) == 0 && owner._delaunay))
    return walk_straight_tet(owner, site, hint);
  auto from = Owner::none;
  for (auto tet = hint;;) {
    const auto next = step_tet_walk(owner, tet, from, site);
    if (next == Owner::none)
      return tet;
    from = tet;
    tet = next;
  }
}

template <typename Owner>
auto locate_conflict_tet(const Owner &owner, typename Owner::index_type site) ->
    typename Owner::index_type {
  return locate_conflict_tet(owner, site, owner._hint);
}

} // namespace tf::topology::cdt::dt3
