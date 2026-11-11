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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Split row `k` at `site` into the reserved row `next`: the row keeps its
/// first part and `next` takes the second, both due for the next round.
template <typename Index, typename Coord>
auto split_tet_subsegment(tet_segment_recovery<Index, Coord> &recovery, Index k,
                          Index next, Index site, std::int64_t numerator)
    -> void {
  auto &row = recovery.subsegments[std::size_t(k)];
  recovery.subsegments[std::size_t(next)] = {
      row.segment, site,   row.b,    row.next,
      numerator,   row.n1, row.hint, row.halvings + 1};
  recovery.proposed[std::size_t(next)] = Index(-1);
  row.b = site;
  row.next = next;
  row.n1 = numerator;
  ++row.halvings;
  auto &last = recovery.last[std::size_t(row.segment)];
  if (last == k)
    last = next;
}

} // namespace tf::topology::cdt::dt3
