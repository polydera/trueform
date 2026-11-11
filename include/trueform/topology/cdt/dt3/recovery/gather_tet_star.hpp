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
#include "../tet_round_state.hpp"
#include "../../../../core/config/verify.hpp"
#include "./tet_star_stamps.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Append to `star` the finite cells of the published complex holding
/// `site`, reached from `from`, one of them, across the facets that hold
/// the site. Each cell is admitted once through its mark at the site's
/// corner in `stamps`, so a star of any size costs its own cells; the
/// generation must not have gathered `site` before. EVERY CELL IS ADMITTED
/// BEFORE ANY LANE OF IT IS READ (@ref admit_tet_row): a cell of another
/// region stops the task, and the star it leaves is partial — the task is
/// deferred and the frontier gathers it whole.
template <typename DT, typename Star, typename Index>
auto gather_tet_star(const DT &dt, typename DT::index_type site,
                     typename DT::index_type from, tet_star_stamps &stamps,
                     Star &star, tet_round_state<Index> &round) -> void {
  const auto cells = dt.tets();
  const auto neighbors = dt.neighbors();
  const auto generation = stamps.generation;
  const auto admit = [&cells, &stamps, &star, &round, site,
                      generation](typename DT::index_type cell) {
    const auto corners = cells[std::size_t(cell)];
    if (!admit_tet_site_patch(round, Index(corners[0])))
      return;
    std::size_t j = 0;
    while (j < 4 && corners[j] != site)
      ++j;
    if constexpr (tf::core::verification)
      tf::core::verify(j < 4, "a star's cell holds its site");
    auto &mark = stamps.marks[4 * std::size_t(cell) + j];
    if (mark == generation)
      return;
    mark = generation;
    star.push_back(cell);
  };
  std::size_t head = star.size();
  admit(from);
  for (; head < star.size() && !round.stopped; ++head) {
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

/// The same outside a round, where every cell is the one task's.
template <typename DT, typename Star>
auto gather_tet_star(const DT &dt, typename DT::index_type site,
                     typename DT::index_type from, tet_star_stamps &stamps,
                     Star &star) -> void {
  tet_round_state<typename DT::index_type> closed;
  gather_tet_star(dt, site, from, stamps, star, closed);
}

} // namespace tf::topology::cdt::dt3
