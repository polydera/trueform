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
#include "./advance_tet_visit.hpp"
#include "./tet_edge_query_workspace.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename DT, typename Index>
auto tet_star_contains_edge(const DT &dt, Index a, Index b, Index from,
                            tet_edge_query_scratch<Index> &scratch) -> bool {
  const auto stamp =
      advance_tet_visit(scratch.visited, scratch.epoch, dt.n_tets());
  scratch.stack.clear();
  scratch.stack.push_back(from);
  scratch.visited[std::size_t(from)] = stamp;
  const auto cells = dt.tets();
  const auto neighbors = dt.neighbors();
  while (!scratch.stack.empty()) {
    const auto cell = scratch.stack.back();
    scratch.stack.pop_back();
    auto corners = cells[std::size_t(cell)];
    for (auto v : corners)
      if (v == b)
        return true;
    for (std::size_t j = 0; j < 4; ++j) {
      if (corners[j] == a)
        continue;
      const auto next = neighbors[std::size_t(cell)][j];
      if (next == DT::k_none || scratch.visited[std::size_t(next)] == stamp)
        continue;
      scratch.visited[std::size_t(next)] = stamp;
      scratch.stack.push_back(next);
    }
  }
  return false;
}

} // namespace tf::topology::cdt::dt3
