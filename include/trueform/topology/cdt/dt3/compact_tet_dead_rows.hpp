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
#include "./compact_tetrahedralization.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Compact canonically once the dead cells reach the owner's share of its
/// rows, and not before: a count decides it, so it falls at the same point
/// at any worker count, and between compactions a reader tolerates dead
/// rows. Returns whether it compacted.
template <bool Parallel, typename Owner>
auto compact_tet_dead_rows(Owner &owner) -> bool {
  if (owner._n_dead * Owner::compaction_dead_divisor <
      std::size_t(owner._corners.size()))
    return false;
  compact_tetrahedralization<Parallel, true>(owner);
  return true;
}

} // namespace tf::topology::cdt::dt3
