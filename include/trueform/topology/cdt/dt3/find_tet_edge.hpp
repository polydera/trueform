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
#include "../../../core/small_vector.hpp"
#include "./find_tet_direction.hpp"
#include "./tet_edge_ref.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// A handle at `site`'s own cell, the site its org.
template <typename Owner>
auto tet_site_handle(const Owner &owner, typename Owner::index_type site)
    -> tet_edge_ref<typename Owner::index_type> {
  const auto cell = owner._site_cells.load(std::size_t(site));
  assert(owner._states[std::size_t(cell)] == tet_state::live);
  const auto corners = owner._corners[std::size_t(cell)];
  return tet_edge_of(owner, cell, site, corners[corners[0] == site ? 1 : 0]);
}

/// Find the edge from `e1` to `e2`: the path from
/// either end, then the cells about `e1`. Returns whether it stands, `edge`
/// then a handle with `e1` its org.
template <typename Owner>
auto find_tet_edge(const Owner &owner, typename Owner::index_type e1,
                   typename Owner::index_type e2,
                   tet_edge_ref<typename Owner::index_type> &edge) -> bool {
  using Index = typename Owner::index_type;
  edge = tet_site_handle(owner, e1);
  if (find_tet_direction(owner, edge, e2) == tet_direction::across_vertex &&
      tet_edge_corners(owner, edge)[1] == e2)
    return true;
  edge = tet_site_handle(owner, e2);
  if (find_tet_direction(owner, edge, e1) == tet_direction::across_vertex &&
      tet_edge_corners(owner, edge)[1] == e1) {
    edge = tet_edge_sym(edge);
    return true;
  }
  tf::small_vector<Index, 64> star;
  star.push_back(owner._site_cells.load(std::size_t(e1)));
  for (std::size_t at = 0; at < star.size(); ++at) {
    const Index cell = star[at];
    const auto corners = owner._corners[std::size_t(cell)];
    for (std::size_t s = 0; s < 4; ++s) {
      if (corners[s] == e2) {
        edge = tet_edge_of(owner, cell, e1, e2);
        return true;
      }
      if (corners[s] == e1)
        continue;
      const Index across = owner._neighbors[std::size_t(cell)][s];
      if (owner._states[std::size_t(across)] != tet_state::live)
        continue;
      if (std::find(star.begin(), star.end(), across) == star.end())
        star.push_back(across);
    }
  }
  return false;
}

} // namespace tf::topology::cdt::dt3
