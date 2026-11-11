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
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Ticket row `row` at the lower of its two sites, first among the rows
/// ticketed there. A site holds exactly the rows whose lower site it is, so
/// a row whose lower site changes is withdrawn first
/// (@ref withdraw_tet_subsegment).
template <typename Index, typename Coord>
auto ticket_tet_subsegment(tet_segment_recovery<Index, Coord> &recovery,
                           Index row) -> void {
  const auto &piece = recovery.subsegments[std::size_t(row)];
  const auto site = std::size_t(std::min(piece.a, piece.b));
  auto &heads = recovery.row_heads;
  if (site >= heads.size()) {
    const std::size_t known = heads.size();
    heads.reallocate(site + 1);
    for (std::size_t s = known; s < heads.size(); ++s)
      heads[s] = Index(-1);
  }
  auto &links = recovery.row_links;
  const Index head = heads[site];
  links[std::size_t(row)] = {Index(-1), head};
  if (head != Index(-1))
    links[std::size_t(head)].previous = row;
  heads[site] = row;
}

} // namespace tf::topology::cdt::dt3
