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
#include "./tet_edge_query_workspace.hpp"
#include "./tet_facets.hpp"
#include "./tet_star_contains_edge.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether the published complex holds the edge `a b`: a walk from `from`,
/// a cell holding `a`, through `a`'s star toward `b`, settled over the whole
/// star when the walk does not arrive within its step bound.
template <typename DT, typename Index>
auto tet_contains_edge(const DT &dt, Index a, Index b, Index from,
                       tet_edge_query_scratch<Index> &scratch) -> bool {
  if (a == b || from == DT::k_none)
    return false;
  const auto sites = dt.sites();
  const auto cells = dt.tets();
  const auto neighbors = dt.neighbors();
  auto cell = from;
  for (std::size_t step = 0; step < 64; ++step) {
    const auto corners = cells[std::size_t(cell)];
    for (auto v : corners)
      if (v == b)
        return true;
    Index next = DT::k_none;
    for (std::size_t j = 0; j < 4; ++j) {
      if (corners[j] == a)
        continue;
      const auto face = tet_facet(corners, j);
      if (tf::exact::orient3d_value_scaled<typename DT::int_type>(
              sites[face[0]].pt, sites[face[1]].pt, sites[face[2]].pt,
              sites[b].pt) > 0) {
        next = neighbors[std::size_t(cell)][j];
        break;
      }
    }
    if (next == DT::k_none)
      return false;
    cell = next;
  }
  return tet_star_contains_edge(dt, a, b, from, scratch);
}

} // namespace tf::topology::cdt::dt3
