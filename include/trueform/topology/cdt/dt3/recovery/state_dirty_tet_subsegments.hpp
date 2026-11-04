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
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The rows the next round reads: the ones this round split, withdrew or
/// deferred (`revisits`), and every pending row whose edge a cavity of this
/// round took, read off the complex's own record of the edges its appends
/// took, each found through its sites' row tickets.
template <typename DT, typename Index, typename Coord>
auto state_dirty_tet_subsegments(DT &dt,
                                 tet_segment_recovery<Index, Coord> &recovery)
    -> void {
  auto &dirty = recovery.dirty;
  dirty.clear();
  const auto pending = [&recovery](Index k) {
    return recovery.status[std::size_t(
               recovery.subsegments[std::size_t(k)].segment)] ==
           tet_segment_status::pending;
  };
  for (auto k : recovery.revisits)
    if (pending(k))
      dirty.push_back(k);
  const auto &killed = dt.killed_edges();
  const auto &heads = recovery.row_heads;
  for (const auto &edge : killed) {
    if (std::size_t(edge[0]) >= heads.size())
      continue;
    for (Index t = heads[std::size_t(edge[0])]; t != Index(-1);
         t = recovery.row_tickets[std::size_t(t)].second) {
      const Index k = recovery.row_tickets[std::size_t(t)].first;
      const auto &row = recovery.subsegments[std::size_t(k)];
      if (((row.a == edge[0] && row.b == edge[1]) ||
           (row.a == edge[1] && row.b == edge[0])) &&
          pending(k))
        dirty.push_back(k);
    }
  }
  std::sort(dirty.begin(), dirty.end());
  dirty.erase_till_end(std::unique(dirty.begin(), dirty.end()));
  recovery.revisits.clear();
  dt.clear_killed_edges();
}

} // namespace tf::topology::cdt::dt3
