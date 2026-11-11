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
#include "../../../core/buffer.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include <atomic>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Release the cells an attempt holds but those whose word carries a bit of
/// `kept`: a completed star keeps its cavity's, the dead staying in the
/// stream's pool and the reused published with the other rows, so no row is
/// released twice.
template <typename Index>
inline auto
release_tet_cells(tet_claim_workspace<Index> &work, std::size_t worker,
                  const tf::buffer<Index> &cells, tet_claim_word kept) -> void {
  for (auto cell : cells) {
    auto &word = work.ownership[std::size_t(cell)];
    const auto held = word.load(std::memory_order_relaxed);
    if ((held & tet_worker_mask) == worker && !(held & kept))
      word.store(tet_unowned, std::memory_order_release);
  }
}

} // namespace tf::topology::cdt::dt3
