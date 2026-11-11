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
#include "../../../core/config/verify.hpp"
#include "./order_tet_cavity_records.hpp"
#include "./tet_cavity_links.hpp"
#include "./tet_execution_tuning.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

template <typename Index>
auto close_tet_cavity_links(tet_cavity_links<Index> &links) -> void {
  links.pending = false;
  const std::size_t n = links.n_facets;
  if (!links.overflowed) {
    if constexpr (tf::core::verification)
      for (std::size_t i = 0; i < n; ++i)
        for (const Index corner : links.facets[i]) {
          const std::size_t local =
              links.local_of_site[tet_cavity_stamp_slot(corner)];
          tf::core::verify(local < links.n_local &&
                               links.site_of_local[local] == corner,
                           "every boundary corner reads its own local id");
        }
    for (std::size_t x = 0; x < 3 * n; ++x)
      if constexpr (tf::core::verification)
        tf::core::verify(links.across[x] >= 0,
                         "every boundary edge of a cavity pairs two facets");
    if constexpr (tf::core::verification) {
      constexpr std::size_t max_local = tet_cavity_links<Index>::max_local;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < 3; ++k) {
          const std::size_t a =
              links.local_of_site[tet_cavity_stamp_slot(links.facets[i][k])];
          const std::size_t b = links.local_of_site[tet_cavity_stamp_slot(
              links.facets[i][k == 2 ? 0 : k + 1])];
          tf::core::verify(
              links.table[std::min(a, b) * max_local + std::max(a, b)] ==
                  Index(-1),
              "a closed cavity leaves the table empty");
        }
    }
    return;
  }
  order_tet_cavity_records(links);
  // An edge's two records share both corners: equal lower corners are one
  // run, and inside it the mate is the record of the same upper corner.
  auto &records = links.records;
  for (std::size_t first = 0; first < records.size();) {
    std::size_t last = first + 1;
    while (last < records.size() && records[last][0] == records[first][0])
      ++last;
    if (last - first > tet_execution_tuning::cavity_run_scan)
      std::sort(records.begin() + std::ptrdiff_t(first),
                records.begin() + std::ptrdiff_t(last),
                [](const auto &x, const auto &y) { return x[1] < y[1]; });
    else
      for (std::size_t r = first; r + 1 < last; r += 2) {
        std::size_t mate = r + 1;
        while (mate + 1 < last && records[mate][1] != records[r][1])
          ++mate;
        std::swap(records[r + 1], records[mate]);
      }
    first = last;
  }
  for (std::size_t r = 0; r + 1 < links.records.size(); r += 2) {
    const auto &first = links.records[r], &second = links.records[r + 1];
    if constexpr (tf::core::verification)
      tf::core::verify(first[0] == second[0] && first[1] == second[1],
                       "every boundary edge of a cavity pairs two facets");
    links.across[std::size_t(first[2])] = second[2];
    links.across[std::size_t(second[2])] = first[2];
  }
}

} // namespace tf::topology::cdt::dt3
