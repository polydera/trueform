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
#include "./tet_state.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Walk from `hint` to a finite cell holding the site or a hull cell seeing
/// it. The walk ends on a Delaunay complex; once facets are protected, or a
/// flip has run (`_delaunay`), the complex is not, a walk can cycle, and it
/// is then bounded by the cell count and the live cells are read in row
/// order after it.
template <typename Owner>
auto locate_conflict_tet(const Owner &owner, typename Owner::index_type site,
                         typename Owner::index_type hint) ->
    typename Owner::index_type {
  using Index = typename Owner::index_type;
  auto from = Owner::none;
  if (owner._n_protected == 0 && owner._delaunay)
    for (auto tet = hint;;) {
      const auto next = step_tet_walk(owner, tet, from, site);
      if (next == Owner::none)
        return tet;
      from = tet;
      tet = next;
    }
  auto tet = hint;
  for (std::size_t step = 0; step <= std::size_t(owner._corners.size()); ++step) {
    const auto next = step_tet_walk(owner, tet, from, site);
    if (next == Owner::none)
      return tet;
    from = tet;
    tet = next;
  }
  for (std::size_t cell = 0; cell < std::size_t(owner._corners.size()); ++cell)
    if (owner._states[cell] != tet_state::dead &&
        step_tet_walk(owner, Index(cell), Owner::none, site) == Owner::none)
      return Index(cell);
  return hint;
}

template <typename Owner>
auto locate_conflict_tet(const Owner &owner, typename Owner::index_type site) ->
    typename Owner::index_type {
  return locate_conflict_tet(owner, site, owner._hint);
}

} // namespace tf::topology::cdt::dt3
