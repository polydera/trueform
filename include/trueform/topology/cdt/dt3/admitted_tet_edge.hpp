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
#include "./tet_edge_ref.hpp"
#include <cstddef>
#include "./tet_round_state.hpp"

namespace tf::topology::cdt::dt3 {

/// The spin, admitted: a task obtains the next cell about an edge THROUGH A
/// LINK, so a cell of another region stops it (@ref admit_tet_row) and the
/// caller leaves its work where it stands. Outside a round these are the
/// spin itself.
template <typename Owner>
auto admitted_tet_edge_next(
    const Owner &owner,
    tet_round_state<typename Owner::index_type> &round,
    const tet_edge_ref<typename Owner::index_type> &edge)
    -> tet_edge_ref<typename Owner::index_type> {
  const auto next = tet_edge_next(owner, edge);
  admit_tet_row(owner, round, next.cell);
  return next;
}

/// The same across the edge's own facet.
template <typename Owner>
auto admitted_tet_edge_across(
    const Owner &owner,
    tet_round_state<typename Owner::index_type> &round,
    const tet_edge_ref<typename Owner::index_type> &edge)
    -> tet_edge_ref<typename Owner::index_type> {
  const auto across = tet_edge_across(owner, edge);
  admit_tet_row(owner, round, across.cell);
  return across;
}

/// Admit every row a flip of `cell` may reach: the cell AND the mate of each
/// of its facets. The kernel's own undo of a ring flip reaches the mate
/// outside the ring (@ref flip_tet_ring's `worse`), so a flip whose closure
/// is not the task's must be refused BEFORE it writes — a stop after a write
/// would leave the ring half flipped and the undo with nothing to undo.
template <typename Owner, typename Index>
auto admit_tet_flip_closure(const Owner &owner,
                            tet_round_state<Index> &round, Index cell)
    -> bool {
  if (!admit_tet_row(owner, round, cell))
    return false;
  for (std::size_t slot = 0; slot < 4; ++slot)
    if (!admit_tet_row(owner, round, owner._neighbors[std::size_t(cell)][slot]))
      return false;
  return true;
}

} // namespace tf::topology::cdt::dt3
