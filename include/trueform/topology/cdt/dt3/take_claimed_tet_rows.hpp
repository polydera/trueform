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
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_state.hpp"
#include <atomic>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Take a row for every boundary facet, the cavity's rows freed first; the
/// caller has proved they suffice. A freed row is dead from here and stays
/// held by the stream, as every row of its pool is, so no other stream
/// reaches it; a taken row is held by this stream alone until the caller
/// publishes the complete star.
template <typename Owner, typename Index>
auto take_claimed_tet_rows(Owner &owner, tet_claim_workspace<Index> &work,
                           std::size_t worker,
                           tet_claim_scratch<Index> &scratch) -> void {
  auto &stream = work.streams[worker];
  auto head = stream.free_head;
  for (auto cell : scratch.cavity) {
    work.free_next[std::size_t(cell)] = head;
    head = cell;
    owner._states[std::size_t(cell)] = tet_state::dead;
  }
  scratch.rows.allocate(scratch.boundary.size());
  std::size_t freed = scratch.cavity.size();
  for (auto &cell : scratch.rows) {
    if (freed) {
      cell = head;
      head = work.free_next[std::size_t(cell)];
      --freed;
      continue;
    }
    if (stream.fresh_begin < stream.fresh_end) {
      cell = Index(stream.fresh_begin++);
      continue;
    }
    cell = head;
    head = work.free_next[std::size_t(cell)];
    // A row of an earlier cavity, or one the barrier moved here naming the
    // stream it came from: it names this one until the star is published.
    work.ownership[std::size_t(cell)].store(tet_claim_word(worker),
                                            std::memory_order_relaxed);
  }
  stream.free_head = head;
  stream.free_count += scratch.cavity.size();
  stream.free_count -= scratch.rows.size();
}

} // namespace tf::topology::cdt::dt3
