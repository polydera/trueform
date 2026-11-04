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
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether the row `k` of `recovery` stands in `dt` as it is now: its second
/// end is in the star of its first, gathered into `recovery.star`, and it is
/// not redirected.
template <typename DT, typename Index, typename Coord>
auto tet_subsegment_stands(const DT &dt, Index k,
                           tet_segment_recovery<Index, Coord> &recovery)
    -> bool {
  const auto &row = recovery.subsegments[std::size_t(k)];
  if (std::binary_search(recovery.redirected.begin(), recovery.redirected.end(),
                         std::array<Index, 2>{k, row.b}))
    return false;
  auto &star = recovery.star;
  star.clear();
  open_tet_star_stamps(dt, recovery.star_stamps);
  bool ticketed = false;
  gather_tet_star(dt, row.a, enter_tet_site_star(dt, row.a, row.hint, ticketed),
                  recovery.star_stamps, star);
  const auto cells = dt.tets();
  for (auto cell : star)
    for (auto corner : cells[std::size_t(cell)])
      if (corner == row.b)
        return true;
  return false;
}

} // namespace tf::topology::cdt::dt3
