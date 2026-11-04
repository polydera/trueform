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
#include "./tet_orient3d_sites.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The cell a walk toward the site `site` takes from the live cell `tet`, or
/// `Owner::none` where the walk ends: in a finite cell that holds it, its
/// facets included, or in a hull cell that sees it. The facet shared with
/// `from`, the cell the walk arrived from, already has it on its inner side.
template <typename Owner>
auto step_tet_walk(const Owner &owner, typename Owner::index_type tet,
                   typename Owner::index_type from,
                   typename Owner::index_type site) ->
    typename Owner::index_type {
  const auto corners = owner._corners[std::size_t(tet)];
  const auto neighbors = owner._neighbors[std::size_t(tet)];
  if (corners[3] == Owner::infinite)
    return tet_orient3d_sites(owner, corners[0], corners[1], corners[2], site) >
                   0
               ? Owner::none
               : neighbors[3];
  for (std::size_t slot = 0; slot < 4; ++slot) {
    if (neighbors[slot] == from)
      continue;
    const auto facet = tet_facet(corners, slot);
    if (tet_orient3d_sites(owner, facet[0], facet[1], facet[2], site) > 0)
      return neighbors[slot];
  }
  return Owner::none;
}

} // namespace tf::topology::cdt::dt3
