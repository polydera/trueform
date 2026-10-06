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
#include "./tet_cell_key.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tbb/parallel_sort.h>

namespace tf::topology::cdt::dt3 {

/// Rewrite the compaction of every finite cell to its rank among the finite
/// cells ordered by their ascending corner slots. Distinct cells hold
/// distinct corner sets, so the order is total.
template <bool Parallel, typename Owner>
auto rank_tet_cells(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  const std::size_t n = owner._corners.size();
  owner._cell_keys.allocate(owner._n_finite);
  const auto key_cell = [&owner](std::size_t i) {
    if (owner._states[i] != tet_state::live)
      return;
    const auto corners = owner._corners[i];
    std::array<Index, 4> key{corners[0], corners[1], corners[2], corners[3]};
    std::sort(key.begin(), key.end());
    owner._cell_keys[std::size_t(owner._compaction[i])] = {key, Index(i)};
  };
  const auto rank_cell = [&owner](std::size_t rank) {
    owner._compaction[std::size_t(owner._cell_keys[rank].row)] = Index(rank);
  };
  const auto less = [](const tet_cell_key<Index> &a,
                       const tet_cell_key<Index> &b) {
    return a.corners < b.corners;
  };
  if constexpr (Parallel) {
    tf::parallel_for_each(tf::make_sequence_range(n), key_cell, tf::checked);
    tbb::parallel_sort(owner._cell_keys.begin(), owner._cell_keys.end(), less);
    tf::parallel_for_each(tf::make_sequence_range(owner._n_finite), rank_cell,
                          tf::checked);
  } else {
    for (std::size_t i = 0; i < n; ++i)
      key_cell(i);
    std::sort(owner._cell_keys.begin(), owner._cell_keys.end(), less);
    for (std::size_t rank = 0; rank < owner._n_finite; ++rank)
      rank_cell(rank);
  }
}

} // namespace tf::topology::cdt::dt3
