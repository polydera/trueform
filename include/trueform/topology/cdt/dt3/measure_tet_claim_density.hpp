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
#include "../../../core/config/verify.hpp"
#include "./tet_claim_workspace.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The complex's live rows per site at a barrier, `inserted` the sites it
/// holds: every row the lanes hold less the dead ones, the pools' among them,
/// and less what the streams' insertions killed beyond what they took.
template <typename Owner, typename Index>
auto measure_tet_claim_density(const Owner &owner,
                               const tet_claim_workspace<Index> &work,
                               std::size_t inserted) -> double {
  long long live = static_cast<long long>(owner._corners.size()) -
                   static_cast<long long>(owner._n_dead);
  for (const auto &stream : work.streams)
    live -= stream.dead_delta;
  if constexpr (tf::core::verification)
    tf::core::verify(inserted > 0 && live > 0,
                     "a barrier stands on a complex holding sites and cells");
  return double(live) / double(inserted);
}

} // namespace tf::topology::cdt::dt3
