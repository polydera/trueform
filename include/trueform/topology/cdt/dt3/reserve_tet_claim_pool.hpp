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
#include "./prepare_tet_claim_pool.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_execution_tuning.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Size the claim pool for `sites` insertions beside the live cells, never
/// past the cell space, and split it over the streams `workers` run.
template <typename Owner, typename Index>
auto reserve_tet_claim_pool(Owner &owner, tet_claim_workspace<Index> &work,
                            std::size_t workers, std::size_t sites) -> void {
  using Tuning = tet_execution_tuning;
  const std::size_t live = owner._corners.size();
  const std::size_t available = std::size_t(Owner::max_tets) - live;
  const auto rows = std::min(available / Tuning::reserve_per_site, sites) *
                    Tuning::reserve_per_site;
  const auto streams = std::min(workers, std::size_t(tet_worker_mask) /
                                             Tuning::streams_per_worker) *
                       Tuning::streams_per_worker;
  prepare_tet_claim_pool(owner, work, streams, live + rows);
}

} // namespace tf::topology::cdt::dt3
