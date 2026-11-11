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
#include "../tet_entry_cell.hpp"
#include "../tet_facets.hpp"
#include "../tet_finite_cell.hpp"
#include "../tet_orient3d_sites.hpp"
#include "../tet_position.hpp"
#include "../tet_protected_count.hpp"
#include "../walk_straight_tet.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// A finite cell of the complex whose closed volume holds the position `p`,
/// walking from the cell `from`, a row it may not enter replaced by the
/// owner's entry cell; `k_none` when `p` lies beyond the hull. On a Delaunay
/// complex the walk steps across a facet `p` lies strictly beyond; otherwise
/// it follows the segment to `p` (@ref walk_straight_tet), which ends in any
/// triangulation.
template <typename DT>
auto walk_to_tet_point(const DT &dt,
                       const tet_position<typename DT::coord_type> &p,
                       typename DT::index_type from) ->
    typename DT::index_type {
  const auto &owner = dt.owner();
  auto cell = tet_finite_cell(dt, from) ? from : tet_entry_cell(dt);
  if (!(tet_protected_count(owner) == 0 && owner._delaunay)) {
    cell = walk_straight_tet(owner, p, cell);
    return tet_finite_cell(dt, cell) ? cell : DT::k_none;
  }
  const auto cells = dt.tets();
  const auto neighbors = dt.neighbors();
  const auto beyond = [&owner, &cells, &p](std::size_t at, std::size_t j) {
    const auto face = tet_facet(cells[at], j);
    return tet_orient3d_toward(owner, face[0], face[1], face[2], p) > 0;
  };
  while (cell != DT::k_none && tet_finite_cell(dt, cell)) {
    std::size_t j = 0;
    while (j < 4 && !beyond(std::size_t(cell), j))
      ++j;
    if (j == 4)
      return cell;
    cell = neighbors[std::size_t(cell)][j];
  }
  return DT::k_none;
}

} // namespace tf::topology::cdt::dt3
