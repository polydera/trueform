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
#include "../../../exact/orient3d.hpp"
#include "./tet_facets.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The cell a walk toward `p` takes from the live cell `tet`, or
/// `Owner::none` where the walk ends: in a finite cell that holds `p`, its
/// facets included, or in a hull cell that sees it. The facet shared with
/// `from`, the cell the walk arrived from, already has `p` on its inner side.
template <typename Owner, typename Point>
auto step_tet_walk(const Owner &owner, typename Owner::index_type tet,
                   typename Owner::index_type from, const Point &p) ->
    typename Owner::index_type {
  using Int = typename Owner::int_type;
  const auto corners = owner._corners[std::size_t(tet)];
  const auto neighbors = owner._neighbors[std::size_t(tet)];
  if (corners[3] == Owner::infinite)
    return tf::exact::orient3d_value_scaled<Int>(
               owner._sites[std::size_t(corners[0])].pt,
               owner._sites[std::size_t(corners[1])].pt,
               owner._sites[std::size_t(corners[2])].pt, p) > 0
               ? Owner::none
               : neighbors[3];
  for (std::size_t slot = 0; slot < 4; ++slot) {
    if (neighbors[slot] == from)
      continue;
    const auto facet = tet_facet(corners, slot);
    if (tf::exact::orient3d_value_scaled<Int>(
            owner._sites[std::size_t(facet[0])].pt,
            owner._sites[std::size_t(facet[1])].pt,
            owner._sites[std::size_t(facet[2])].pt, p) > 0)
      return neighbors[slot];
  }
  return Owner::none;
}

} // namespace tf::topology::cdt::dt3
