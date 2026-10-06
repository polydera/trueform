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
#include "./tet_facets.hpp"
#include "./write_tet.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Close a seed tetrahedron with one hull cell per facet.
template <typename Owner>
auto close_seed_tet(Owner &owner, typename Owner::index_type seed) -> void {
  using Index = typename Owner::index_type;
  const auto first = std::size_t(owner._corners.size());
  owner._corners.reallocate(first + 4);
  owner._neighbors.reallocate(first + 4);
  owner._states.reallocate(first + 4);
  const auto corners = owner._corners[std::size_t(seed)];
  const std::array<Index, 4> slots{0, 1, 2, 3};
  for (std::size_t side = 0; side < 4; ++side) {
    const auto facet = tet_facet(corners, side);
    const auto opposite = tet_facet(slots, side);
    const auto hull = Index(first + side);
    write_tet(owner, hull, facet[0], facet[1], facet[2], Owner::infinite);
    owner._neighbors[std::size_t(seed)][side] = hull;
    auto links = owner._neighbors[first + side];
    links[3] = seed;
    for (std::size_t corner = 0; corner < 3; ++corner)
      links[corner] = Index(first + std::size_t(opposite[corner]));
  }
}

} // namespace tf::topology::cdt::dt3
