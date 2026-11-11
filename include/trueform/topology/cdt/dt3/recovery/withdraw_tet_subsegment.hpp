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
#include "../../../../core/config/verify.hpp"
#include "./tet_segment_recovery.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Withdraw row `row` from the rows ticketed at `site`, the lower site it
/// was ticketed at (@ref ticket_tet_subsegment).
template <typename Index, typename Coord>
auto withdraw_tet_subsegment(tet_segment_recovery<Index, Coord> &recovery,
                             Index site, Index row) -> void {
  auto &links = recovery.row_links;
  const auto link = links[std::size_t(row)];
  if constexpr (tf::core::verification)
    tf::core::verify(link.previous != Index(-1)
                         ? links[std::size_t(link.previous)].next == row
                         : recovery.row_heads[std::size_t(site)] == row,
                     "a withdrawn row is ticketed at its lower site");
  if (link.previous != Index(-1))
    links[std::size_t(link.previous)].next = link.next;
  else
    recovery.row_heads[std::size_t(site)] = link.next;
  if (link.next != Index(-1))
    links[std::size_t(link.next)].previous = link.previous;
}

} // namespace tf::topology::cdt::dt3
