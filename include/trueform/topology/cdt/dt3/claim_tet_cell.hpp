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
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Index>
inline auto claim_tet_cell(tet_claim_workspace<Index> &work, std::size_t worker,
                           tet_claim_scratch<Index> &scratch, Index cell,
                           tet_claim_word &word) -> bool {
  word = work.ownership[std::size_t(cell)].load(std::memory_order_relaxed);
  if ((word & tet_worker_mask) == worker && !(word & tet_pooled))
    return true;
  if (!acquire_tet_cell(work, worker, cell, scratch.interfering)) {
    scratch.interfering_cell = cell;
    return false;
  }
  scratch.held.push_back(cell);
  word = tet_claim_word(worker);
  return true;
}

} // namespace tf::topology::cdt::dt3
