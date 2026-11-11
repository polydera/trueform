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

/// The row running between the sites `p` and `q` of a segment still a
/// constraint, read among the rows ticketed at the lower of the two, `-1`
/// where none does.
template <typename Index, typename Coord>
auto tet_piece_row(const tet_segment_recovery<Index, Coord> &recovery, Index p,
                   Index q) -> Index {
  const Index low = std::min(p, q), high = std::max(p, q);
  const auto &heads = recovery.row_heads;
  if (std::size_t(low) >= heads.size())
    return Index(-1);
  for (Index k = heads[std::size_t(low)]; k != Index(-1);
       k = recovery.row_links[std::size_t(k)].next) {
    const auto &row = recovery.subsegments[std::size_t(k)];
    if (std::max(row.a, row.b) == high)
      return k;
  }
  return Index(-1);
}

} // namespace tf::topology::cdt::dt3
