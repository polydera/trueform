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
#include "./acquire_tet_cell.hpp"
#include "./step_tet_walk.hpp"
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include <atomic>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto locate_claimed_tet(const Owner &owner, tet_claim_workspace<Index> &work,
                        std::size_t worker, Index site, Index hint,
                        tet_claim_scratch<Index> &scratch) -> Index {
  if (hint < 0 || !acquire_tet_cell(work, worker, hint, scratch.interfering)) {
    // A stream draws its rows from its own range, so the first free row past
    // the hint lies in the region the stream is filling.
    const std::size_t rows = work.ownership.size();
    const std::size_t from = hint < 0 ? 0 : std::size_t(hint);
    hint = -1;
    for (std::size_t k = 0; k < rows; ++k) {
      const auto i = (from + k) % rows;
      if (work.ownership[i].load(std::memory_order_relaxed) == tet_unowned &&
          acquire_tet_cell(work, worker, Index(i), scratch.interfering)) {
        hint = Index(i);
        break;
      }
    }
    if (hint < 0)
      return -1;
  }
  const auto &p = owner._sites[std::size_t(site)].pt;
  Index previous = Owner::none;
  for (;;) {
    const auto next = step_tet_walk(owner, hint, previous, p);
    if (next == Owner::none) {
      scratch.held.push_back(hint);
      return hint;
    }
    // Hold the source until the destination is claimed: its adjacency cannot
    // become stale.
    const bool acquired =
        acquire_tet_cell(work, worker, next, scratch.interfering);
    work.ownership[std::size_t(hint)].store(tet_unowned,
                                            std::memory_order_release);
    if (!acquired) {
      scratch.interfering_cell = next;
      return -1;
    }
    previous = hint;
    hint = next;
  }
}

} // namespace tf::topology::cdt::dt3
