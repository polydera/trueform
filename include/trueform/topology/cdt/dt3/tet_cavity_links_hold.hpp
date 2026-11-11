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
#include "./tet_cavity_links.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Whether the closed boundary holds the edge `{a, b}`: through the marked
/// table (`mark_tet_cavity_edges`), or the run of its lower corner past an
/// overflow.
template <typename Index>
auto tet_cavity_links_hold(const tet_cavity_links<Index> &links, Index a,
                           Index b) -> bool {
  const Index lo = a < b ? a : b, hi = a < b ? b : a;
  if (links.overflowed) {
    auto record = std::lower_bound(
        links.records.begin(), links.records.end(), lo,
        [](const std::array<Index, 3> &x, Index y) { return x[0] < y; });
    for (; record != links.records.end() && (*record)[0] == lo; ++record)
      if ((*record)[1] == hi)
        return true;
    return false;
  }
  constexpr std::size_t max_local = tet_cavity_links<Index>::max_local;
  const std::size_t la = links.local_of_site[tet_cavity_stamp_slot(a)],
                    lb = links.local_of_site[tet_cavity_stamp_slot(b)];
  if (la >= links.n_local || links.site_of_local[la] != a ||
      lb >= links.n_local || links.site_of_local[lb] != b)
    return false;
  const std::size_t llo = la < lb ? la : lb, lhi = la < lb ? lb : la;
  return links.table[llo * max_local + lhi] == tet_cavity_links<Index>::marked;
}

} // namespace tf::topology::cdt::dt3
