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
#include "./tet_execution_tuning.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <bool Parallel, typename Owner>
auto assign_tet_compaction(Owner &owner) -> std::size_t {
  using Index = typename Owner::index_type;
  const auto n = std::size_t(owner._corners.size());
  owner._compaction.allocate(n);
  if constexpr (!Parallel) {
    Index next = 0;
    for (std::size_t i = 0; i < n; ++i)
      owner._compaction[i] =
          owner._states[i] == tet_state::live ? next++ : Owner::none;
    owner._n_finite = std::size_t(next);
    for (std::size_t i = 0; i < n; ++i)
      if (owner._states[i] == tet_state::infinite)
        owner._compaction[i] = next++;
    return std::size_t(next);
  } else {
    const std::size_t grain = tet_execution_tuning::scan_block_rows;
    const auto blocks = (n + grain - 1) / grain;
    auto &offsets = owner._block_offsets;
    offsets.allocate(2 * (blocks + 1));
    offsets[0] = offsets[blocks + 1] = 0;
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&](std::size_t block) {
          std::size_t finite = 0, hull = 0;
          const auto last = std::min(n, (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i) {
            finite += owner._states[i] == tet_state::live;
            hull += owner._states[i] == tet_state::infinite;
          }
          offsets[block + 1] = finite;
          offsets[blocks + block + 2] = hull;
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
    for (std::size_t block = 0; block < blocks; ++block) {
      offsets[block + 1] += offsets[block];
      offsets[blocks + block + 2] += offsets[blocks + block + 1];
    }
    owner._n_finite = offsets[blocks];
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&](std::size_t block) {
          auto finite = offsets[block];
          auto hull = owner._n_finite + offsets[blocks + block + 1];
          const auto last = std::min(n, (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i) {
            const auto state = owner._states[i];
            owner._compaction[i] = state == tet_state::live ? Index(finite++)
                                   : state == tet_state::infinite
                                       ? Index(hull++)
                                       : Owner::none;
          }
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
    return owner._n_finite + offsets[2 * blocks + 1];
  }
}

} // namespace tf::topology::cdt::dt3
