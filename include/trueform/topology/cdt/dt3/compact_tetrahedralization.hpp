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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./assign_tet_compaction.hpp"
#include "./canonical_tet_slots.hpp"
#include "./rank_tet_cells.hpp"
#include "./tetrahedralization_owner.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Compact finite cells before hull cells, remapping neighbors and the hint;
/// a retired hint falls back to cell zero. A compaction keeps the relative
/// order of its cells; a canonical one publishes the finite cells in
/// @ref rank_tet_cells order, each in its @ref canonical_tet_slots.
template <bool Parallel = false, bool Canonical = false, typename Owner>
auto compact_tetrahedralization(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  const std::size_t n = owner._corners.size();
  const auto live = assign_tet_compaction<Parallel>(owner);
  if constexpr (Canonical)
    rank_tet_cells<Parallel>(owner);
  owner._corner_scratch.allocate(live);
  owner._neighbor_scratch.allocate(live);
  owner._state_scratch.allocate(live);
  const auto copy_cell = [&owner](std::size_t i) {
    const Index to = owner._compaction[i];
    if (to == Owner::none)
      return;
    std::array<std::size_t, 4> slots{0, 1, 2, 3};
    if constexpr (Canonical)
      if (std::size_t(to) < owner._n_finite)
        slots = canonical_tet_slots(owner._corners[i]);
    auto corners = owner._corner_scratch[std::size_t(to)];
    auto neighbors = owner._neighbor_scratch[std::size_t(to)];
    for (std::size_t slot = 0; slot < 4; ++slot) {
      corners[slot] = owner._corners[i][slots[slot]];
      neighbors[slot] =
          owner._compaction[std::size_t(owner._neighbors[i][slots[slot]])];
    }
    owner._state_scratch[std::size_t(to)] = owner._states[i];
  };
  if constexpr (Parallel)
    tf::parallel_for_each(tf::make_sequence_range(n), copy_cell, tf::checked);
  else
    for (std::size_t i = 0; i < n; ++i)
      copy_cell(i);

  std::swap(owner._corners, owner._corner_scratch);
  std::swap(owner._neighbors, owner._neighbor_scratch);
  std::swap(owner._states, owner._state_scratch);
  const Index hint = owner._compaction[std::size_t(owner._hint)];
  owner._hint = hint == Owner::none ? Index(0) : hint;
  owner._n_dead = 0;
  ++owner._stats.compactions;
}

} // namespace tf::topology::cdt::dt3
