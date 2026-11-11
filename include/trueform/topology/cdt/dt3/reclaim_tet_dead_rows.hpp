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

/// Reclaim the dead rows between two operations of a pass, keeping the
/// relative order of the live ones (@ref compact_tetrahedralization), once
/// the dead cells reach the owner's share of its rows and its reclamation
/// floor. The recorded rows lose their dead entries with them, so the record
/// holds the live rows and those written since. A count decides it, so it
/// falls at the same point at any worker count. Returns whether it reclaimed.
template <bool Parallel, typename Owner>
auto reclaim_tet_dead_rows(Owner &owner) -> bool {
  if (owner._n_dead < Owner::reclamation_dead_floor ||
      owner._n_dead * Owner::compaction_dead_divisor <
          std::size_t(owner._corners.size()))
    return false;
  compact_tetrahedralization<Parallel, false>(owner);
  return true;
}

} // namespace tf::topology::cdt::dt3
