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
#include "../../../core/config/verify.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_state.hpp"
#include <atomic>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Give back the rows every stream's pool still holds, its fresh range and
/// its free list, so the lane ends the insertions with every word released;
/// each such row is dead.
template <typename Owner, typename Index>
auto release_tet_claim_pools(const Owner &owner,
                             tet_claim_workspace<Index> &work) -> void {
  tf::parallel_for_each(
      tf::make_sequence_range(work.streams.size()), [&](std::size_t worker) {
        const auto &stream = work.streams[worker];
        std::size_t released = 0;
        const auto release = [&](std::size_t row) {
          if constexpr (tf::core::verification)
            tf::core::verify(owner._states[row] == tet_state::dead,
                             "a row a pool holds is dead");
          work.ownership[row].store(tet_unowned, std::memory_order_relaxed);
          ++released;
        };
        for (auto row = stream.fresh_begin; row < stream.fresh_end; ++row)
          release(row);
        for (auto row = stream.free_head; row >= 0;
             row = work.free_next[std::size_t(row)])
          release(std::size_t(row));
        if constexpr (tf::core::verification)
          tf::core::verify(released == stream.free_count,
                           "a pool holds the rows it counts");
      });
}

} // namespace tf::topology::cdt::dt3
