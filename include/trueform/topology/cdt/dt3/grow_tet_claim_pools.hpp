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
#include "../../../core/views/sequence_range.hpp"
#include "./tet_claim_rows_asked.hpp"
#include "./tet_claim_stream.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// At a barrier, grow the lanes by the rows every stream lacks of its ask at
/// `density` (@ref tet_claim_rows_asked), never past the cell space: each
/// grown stream holds its appended rows as a fresh range, each a dead row,
/// and a fresh range it still held joins its free list. The rows already
/// standing keep their content and their claim words, so a growth touches
/// its own rows only; one that passes the lanes' capacity reserves twice
/// what it appends, untouched, so the barriers after it grow in place.
/// Whether any stream grew.
template <typename Owner, typename Index>
auto grow_tet_claim_pools(Owner &owner, tet_claim_workspace<Index> &work,
                          double density) -> bool {
  const std::size_t live = std::size_t(owner._corners.size());
  std::size_t count = live;
  for (auto &stream : work.streams) {
    const auto asked = tet_claim_rows_asked(stream, density);
    if (stream.free_count >= asked)
      continue;
    const auto grown = std::min(asked - stream.free_count,
                                std::size_t(Owner::max_tets) - count);
    if (!grown)
      break;
    if (stream.fresh_begin < stream.fresh_end) {
      work.free_next[stream.fresh_end - 1] = stream.free_head;
      stream.free_head = Index(stream.fresh_begin);
    }
    stream.fresh_begin = count;
    stream.fresh_end = count + grown;
    stream.free_count += grown;
    count += grown;
  }
  if (count == live)
    return false;
  if (count > owner._states.capacity()) {
    const auto reserved =
        std::min(count + (count - live), std::size_t(Owner::max_tets));
    owner._corners.reserve(reserved);
    owner._neighbors.reserve(reserved);
    owner._states.reserve(reserved);
    work.free_next.reserve(reserved);
    work.ownership.reserve(reserved);
  }
  owner._corners.reallocate(count);
  owner._neighbors.reallocate(count);
  owner._states.reallocate(count);
  owner._n_dead += count - live;
  work.ownership.reallocate_and_initialize(count, tet_unowned);
  work.free_next.reallocate(count);
  tf::parallel_for_each(
      tf::make_sequence_range(work.streams.size()), [&](std::size_t worker) {
        const auto &stream = work.streams[worker];
        if (stream.fresh_begin < live)
          return;
        for (auto i = stream.fresh_begin; i < stream.fresh_end; ++i) {
          work.free_next[i] = i + 1 < stream.fresh_end ? Index(i + 1) : -1;
          work.ownership[i].store(std::uint32_t(worker),
                                  std::memory_order_relaxed);
          owner._states[i] = tet_state::dead;
        }
      });
  return true;
}

} // namespace tf::topology::cdt::dt3
