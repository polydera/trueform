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
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Index>
inline auto acquire_tet_cell(tet_claim_workspace<Index> &work,
                             std::size_t worker, Index cell, int &interfering)
    -> bool {
  auto &word = work.ownership[std::size_t(cell)];
  tet_claim_word expected = tet_unowned;
  if (word.compare_exchange_strong(expected, std::uint32_t(worker),
                                   std::memory_order_acquire,
                                   std::memory_order_relaxed))
    return true;
  if ((expected & tet_worker_mask) == worker && !(expected & tet_pooled))
    return true;
  interfering = (expected & tet_pooled) ? -1 : int(expected & tet_worker_mask);
  return false;
}

} // namespace tf::topology::cdt::dt3
