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
/// table (`mark_tet_cavity_edges`), or the sorted records past an overflow.
template <typename Index>
auto tet_cavity_links_hold(const tet_cavity_links<Index> &links, Index a,
                           Index b) -> bool {
  const Index lo = a < b ? a : b, hi = a < b ? b : a;
  if (links.overflowed) {
    const std::array<Index, 3> key{lo, hi, Index(-1)};
    const auto it = std::lower_bound(
        links.records.begin(), links.records.end(), key,
        [](const auto &x, const auto &y) {
          return x[0] < y[0] || (x[0] == y[0] && x[1] < y[1]);
        });
    return it != links.records.end() && (*it)[0] == lo && (*it)[1] == hi;
  }
  constexpr std::size_t max_local = tet_cavity_links<Index>::max_local;
  const std::uint32_t generation = links.generation << 8;
  const std::uint32_t sa = links.local_of_site[tet_cavity_stamp_slot(a)],
                      sb = links.local_of_site[tet_cavity_stamp_slot(b)];
  if ((sa & ~std::uint32_t(255)) != generation ||
      (sb & ~std::uint32_t(255)) != generation)
    return false;
  const std::uint32_t la = sa & 255u, lb = sb & 255u;
  const std::uint32_t llo = la < lb ? la : lb, lhi = la < lb ? lb : la;
  return links.table[std::size_t(llo) * max_local + std::size_t(lhi)] ==
         tet_cavity_links<Index>::marked;
}

} // namespace tf::topology::cdt::dt3
