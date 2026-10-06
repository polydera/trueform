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
#include "../../../core/views/sequence_range.hpp"
#include "./tet_cavity_edge_table.hpp"
#include "./tet_state.hpp"
#include "./write_tet_cavity.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Grow the cell lanes once and construct the cavity star. Returns the first
/// appended cell, used as the next location hint.
template <typename Owner>
auto star_tet_cavity(Owner &owner, typename Owner::index_type apex,
                     tet_cavity_edge_table<typename Owner::index_type> &edges)
    -> typename Owner::index_type {
  using Index = typename Owner::index_type;
  const Index first = Index(owner._corners.size());
  const std::size_t walls = owner._boundary.size();
  owner._corners.reallocate(std::size_t(first) + walls);
  owner._neighbors.reallocate(std::size_t(first) + walls);
  owner._states.reallocate(std::size_t(first) + walls);
  const auto rows =
      tf::make_sequence_range(std::size_t(first), std::size_t(first) + walls);
  write_tet_cavity(owner, apex, rows, owner._boundary, owner._sides, edges);
  for (auto row : rows)
    owner._states[row] = owner._corners[row][3] == Owner::infinite
                             ? tet_state::infinite
                             : tet_state::live;
  return first;
}

} // namespace tf::topology::cdt::dt3
