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
#include "../tet_facets.hpp"
#include "../tet_finite_cell.hpp"
#include "../tet_orient3d_sites.hpp"
#include "../tet_position.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// A finite cell of the complex whose closed volume holds the position `p`,
/// walking from the cell `from` across a facet `p` lies strictly beyond;
/// `k_none` when `p` lies beyond the hull. A walk from a row it may not enter,
/// or one that outlasts the cell count, reads the finite cells in row order
/// instead.
template <typename DT>
auto walk_to_tet_point(const DT &dt,
                       const tet_position<typename DT::coord_type> &p,
                       typename DT::index_type from) ->
    typename DT::index_type {
  using Index = typename DT::index_type;
  const auto &owner = dt.owner();
  const auto cells = dt.tets();
  const auto neighbors = dt.neighbors();
  const auto beyond = [&owner, &cells, &p](std::size_t cell, std::size_t j) {
    const auto face = tet_facet(cells[cell], j);
    return tet_orient3d_toward(owner, face[0], face[1], face[2], p) > 0;
  };
  Index cell = tet_finite_cell(dt, from) ? from : tet_entry_cell(dt);
  for (std::size_t step = 0; tet_finite_cell(dt, cell) && step <= cells.size();
       ++step) {
    std::size_t j = 0;
    while (j < 4 && !beyond(std::size_t(cell), j))
      ++j;
    if (j == 4)
      return cell;
    cell = neighbors[std::size_t(cell)][j];
    if (cell == DT::k_none)
      return DT::k_none;
  }
  for (std::size_t c = 0; c < cells.size(); ++c) {
    if (!tet_finite_cell(dt, Index(c)))
      continue;
    std::size_t j = 0;
    while (j < 4 && !beyond(c, j))
      ++j;
    if (j == 4)
      return Index(c);
  }
  return DT::k_none;
}

} // namespace tf::topology::cdt::dt3
