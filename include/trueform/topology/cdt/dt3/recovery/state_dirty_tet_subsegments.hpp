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
#include "./revive_tet_segments.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The rows the next round reads: the pending ones this round split, withdrew
/// or deferred (`revisits`), and the rows whose edges a cavity of this round
/// took (@ref revive_tet_segments).
template <typename DT, typename Index, typename Coord>
auto state_dirty_tet_subsegments(DT &dt,
                                 tet_segment_recovery<Index, Coord> &recovery)
    -> void {
  auto &dirty = recovery.dirty;
  dirty.clear();
  for (auto k : recovery.revisits)
    if (recovery.status[std::size_t(
            recovery.subsegments[std::size_t(k)].segment)] ==
        tet_segment_status::pending)
      dirty.push_back(k);
  revive_tet_segments(dt, recovery, dirty);
  std::sort(dirty.begin(), dirty.end());
  dirty.erase_till_end(std::unique(dirty.begin(), dirty.end()));
  recovery.revisits.clear();
}

} // namespace tf::topology::cdt::dt3
