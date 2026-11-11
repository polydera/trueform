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
#include "./tet_piece_row.hpp"
#include "./tet_segment_recovery.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The segment whose row runs between the sites `p` and `q`
/// (@ref tet_piece_row), `-1` where no row of a segment still a constraint
/// does.
template <typename Index, typename Coord>
auto tet_piece_segment(const tet_segment_recovery<Index, Coord> &recovery,
                       Index p, Index q) -> Index {
  const Index k = tet_piece_row(recovery, p, q);
  return k == Index(-1) ? k : recovery.subsegments[std::size_t(k)].segment;
}

} // namespace tf::topology::cdt::dt3
