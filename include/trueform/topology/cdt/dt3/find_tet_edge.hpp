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
#include "../../../core/config/verify.hpp"
#include "./admit_tet_edge_star.hpp"
#include "./find_tet_direction.hpp"
#include "./tet_edge_ref.hpp"
#include "./tet_edge_star.hpp"
#include "./tet_round_state.hpp"
#include "./tet_state.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// A handle at `site`'s own cell, the site its org.
template <typename Owner>
auto tet_site_handle(const Owner &owner, typename Owner::index_type site)
    -> tet_edge_ref<typename Owner::index_type> {
  const auto cell = owner._site_cells.load(std::size_t(site));
  if constexpr (tf::core::verification)
    tf::core::verify(owner._states[std::size_t(cell)] == tet_state::live,
                     "a site's cell is live");
  const auto corners = owner._corners[std::size_t(cell)];
  return tet_edge_of(owner, cell, site, corners[corners[0] == site ? 1 : 0]);
}

/// The cell the site-cell lane answers, admitted: the answer is used only
/// where that cell carries the task's own mark, which is the slot's own law
/// (@ref tet_site_cells); a slot of another region stops the task.
template <typename Owner>
auto admitted_tet_site_handle(
    const Owner &owner, tet_round_state<typename Owner::index_type> &round,
    typename Owner::index_type site)
    -> tet_edge_ref<typename Owner::index_type> {
  using Index = typename Owner::index_type;
  if (round.open() &&
      !admit_tet_row(owner, round, Index(owner._site_cells.load(std::size_t(site)))))
    return tet_edge_ref<Index>{};
  return tet_site_handle(owner, site);
}

/// Find the edge from `e1` to `e2`: the path from either end, then the
/// cells about `e1`, gathered into `star` (@ref tet_edge_star). Returns
/// whether it stands, `edge` then a handle with `e1` its org. EVERY CELL IS
/// ADMITTED BEFORE ANY LANE OF IT IS READ (@ref admit_tet_row), the lane's
/// own answer among them, so a cell of another region stops the task and the
/// edge is not found — the task is deferred, never refused.
template <typename Owner, typename Index>
auto find_tet_edge(const Owner &owner, typename Owner::index_type e1,
                   typename Owner::index_type e2,
                   tet_edge_ref<typename Owner::index_type> &edge,
                   tet_edge_star<typename Owner::index_type> &star,
                   tet_round_state<Index> &round) -> bool {
  edge = admitted_tet_site_handle(owner, round, e1);
  if (round.stopped)
    return false;
  if (find_tet_direction(owner, edge, e2) == tet_direction::across_vertex &&
      tet_edge_corners(owner, edge)[1] == e2)
    return true;
  edge = admitted_tet_site_handle(owner, round, e2);
  if (round.stopped)
    return false;
  if (find_tet_direction(owner, edge, e1) == tet_direction::across_vertex &&
      tet_edge_corners(owner, edge)[1] == e1) {
    edge = tet_edge_sym(edge);
    return true;
  }
  auto &cells = star.cells;
  cells.clear();
  {
    const Index seat = Index(owner._site_cells.load(std::size_t(e1)));
    if (!admit_tet_row(owner, round, seat))
      return false;
    admit_tet_edge_star(star, seat);
  }
  for (std::size_t at = 0; at < cells.size(); ++at) {
    const Index cell = cells[at];
    const auto corners = owner._corners[std::size_t(cell)];
    for (std::size_t s = 0; s < 4; ++s) {
      if (corners[s] == e2) {
        edge = tet_edge_of(owner, cell, e1, e2);
        return true;
      }
      if (corners[s] == e1)
        continue;
      const Index across = owner._neighbors[std::size_t(cell)][s];
      if (!admit_tet_row(owner, round, across))
        return false;
      if (owner._states[std::size_t(across)] != tet_state::live)
        continue;
      admit_tet_edge_star(star, across);
    }
  }
  return false;
}

/// The same outside a round, where every cell is the one task's.
template <typename Owner>
auto find_tet_edge(const Owner &owner, typename Owner::index_type e1,
                   typename Owner::index_type e2,
                   tet_edge_ref<typename Owner::index_type> &edge,
                   tet_edge_star<typename Owner::index_type> &star) -> bool {
  tet_round_state<typename Owner::index_type> closed;
  return find_tet_edge(owner, e1, e2, edge, star, closed);
}

} // namespace tf::topology::cdt::dt3
