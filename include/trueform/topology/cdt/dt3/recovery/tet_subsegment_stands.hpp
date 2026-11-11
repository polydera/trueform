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
#include "./enter_tet_site_star.hpp"
#include "./gather_tet_star.hpp"
#include "./open_tet_star_stamps.hpp"
#include "./tet_segment_recovery.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether the row `k` of `recovery` stands in `dt` as it is now: its second
/// end is in the star of its first, gathered into the flips' own star, which
/// is the region's where a round runs them beside each other.
template <typename DT, typename Index, typename Coord>
auto tet_subsegment_stands(const DT &dt, Index k,
                           tet_segment_recovery<Index, Coord> &recovery)
    -> bool {
  const auto &row = recovery.subsegments[std::size_t(k)];
  auto &star = recovery.flips.star.cells;
  star.clear();
  open_tet_star_stamps(dt, recovery.flips.star_stamps);
  bool ticketed = false;
  gather_tet_star(dt, row.a, enter_tet_site_star(dt, row.a, row.hint, ticketed),
                  recovery.flips.star_stamps, star);
  const auto cells = dt.tets();
  for (auto cell : star)
    for (auto corner : cells[std::size_t(cell)])
      if (corner == row.b)
        return true;
  return false;
}

} // namespace tf::topology::cdt::dt3
