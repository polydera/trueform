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
#include "./tet_conflicts.hpp"
#include "./tet_state.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Mark the conflict cavity dead and record its boundary from surviving cells.
/// A dead neighbor is already visited by this flood.
template <typename Owner>
auto flood_tet_cavity(Owner &owner, typename Owner::index_type seed,
                      typename Owner::index_type site) -> void {
  using Index = typename Owner::index_type;
  auto &cavity = owner._cavity;
  auto &boundary = owner._boundary;
  cavity.clear();
  boundary.clear();
  owner._states[std::size_t(seed)] = tet_state::dead;
  cavity.push_back(seed);
  for (std::size_t head = 0; head < cavity.size(); ++head) {
    const Index cell = cavity[head];
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const Index side = owner._neighbors[std::size_t(cell)][slot];
      if (owner._states[std::size_t(side)] == tet_state::dead)
        continue;
      if (tet_conflicts(owner, side, site, [](Index) { return true; })) {
        owner._states[std::size_t(side)] = tet_state::dead;
        cavity.push_back(side);
        continue;
      }
      std::size_t mate = 0;
      while (owner._neighbors[std::size_t(side)][mate] != cell)
        ++mate;
      boundary.push_back(Index(4 * std::size_t(side) + mate));
    }
  }
  owner._n_dead += cavity.size();
}

} // namespace tf::topology::cdt::dt3
