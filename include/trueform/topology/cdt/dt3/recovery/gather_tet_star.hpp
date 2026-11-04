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
#include "./tet_star_stamps.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Append to `star` the finite cells of the published complex holding
/// `site`, reached from `from`, one of them, across the facets that hold
/// the site. Each cell is admitted once through its mark at the site's
/// corner in `stamps`, so a star of any size costs its own cells; the
/// generation must not have gathered `site` before.
template <typename DT, typename Star>
auto gather_tet_star(const DT &dt, typename DT::index_type site,
                     typename DT::index_type from, tet_star_stamps &stamps,
                     Star &star) -> void {
  const auto cells = dt.tets();
  const auto neighbors = dt.neighbors();
  const auto generation = stamps.generation;
  const auto admit = [&cells, &stamps, &star, site,
                      generation](typename DT::index_type cell) {
    const auto corners = cells[std::size_t(cell)];
    std::size_t j = 0;
    while (corners[j] != site)
      ++j;
    auto &mark = stamps.marks[4 * std::size_t(cell) + j];
    if (mark == generation)
      return;
    mark = generation;
    star.push_back(cell);
  };
  std::size_t head = star.size();
  admit(from);
  for (; head < star.size(); ++head) {
    const auto cell = star[head];
    const auto corners = cells[std::size_t(cell)];
    for (std::size_t j = 0; j < 4; ++j) {
      if (corners[j] == site)
        continue;
      const auto next = neighbors[std::size_t(cell)][j];
      if (next != DT::k_none)
        admit(next);
    }
  }
}

} // namespace tf::topology::cdt::dt3
