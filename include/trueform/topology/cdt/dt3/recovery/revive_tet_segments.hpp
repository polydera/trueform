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
#include "../../../../core/buffer.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The rows the edges the complex `dt` recorded as killed took, each found
/// among the rows ticketed at the edge's lower site, made due: a recovered
/// segment whose row a cavity took turns pending again, and every pending
/// such row is appended to `due`; the record is cleared. The one reader of
/// the killed edges. Returns the segments revived.
template <typename DT, typename Index, typename Coord>
auto revive_tet_segments(DT &dt, tet_segment_recovery<Index, Coord> &recovery,
                         tf::buffer<Index> &due) -> std::size_t {
  std::size_t revived = 0;
  const auto &heads = recovery.row_heads;
  for (const auto &edge : dt.killed_edges()) {
    const Index low = std::min(edge[0], edge[1]);
    const Index high = std::max(edge[0], edge[1]);
    if (std::size_t(low) >= heads.size())
      continue;
    for (Index k = heads[std::size_t(low)]; k != Index(-1);
         k = recovery.row_links[std::size_t(k)].next) {
      const auto &row = recovery.subsegments[std::size_t(k)];
      if (std::max(row.a, row.b) != high)
        continue;
      auto &status = recovery.status[std::size_t(row.segment)];
      if (status == tet_segment_status::recovered) {
        status = tet_segment_status::pending;
        ++revived;
      }
      if (status == tet_segment_status::pending)
        due.push_back(k);
    }
  }
  dt.clear_killed_edges();
  return revived;
}

} // namespace tf::topology::cdt::dt3
