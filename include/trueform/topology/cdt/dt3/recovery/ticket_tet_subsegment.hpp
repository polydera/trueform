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
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Ticket row `row` at `site`: the row is reachable from the site's chain
/// of tickets, newest first. A row that later leaves the site keeps a
/// stale ticket there, which a reader skips by the row's own ends, so a
/// ticket costs O(1) and is never taken back.
template <typename Index, typename Coord>
auto ticket_tet_subsegment(tet_segment_recovery<Index, Coord> &recovery,
                           Index site, Index row) -> void {
  auto &head = recovery.row_heads;
  if (std::size_t(site) >= head.size()) {
    const std::size_t known = head.size();
    head.reallocate(std::size_t(site) + 1);
    for (std::size_t s = known; s < head.size(); ++s)
      head[s] = Index(-1);
  }
  recovery.row_tickets.push_back({row, head[std::size_t(site)]});
  head[std::size_t(site)] = Index(recovery.row_tickets.size() - 1);
}

} // namespace tf::topology::cdt::dt3
