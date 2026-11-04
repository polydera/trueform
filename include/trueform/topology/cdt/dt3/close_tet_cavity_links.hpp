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
#include <cassert>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Index>
auto close_tet_cavity_links(tet_cavity_links<Index> &links) -> void {
  links.pending = false;
  const std::size_t n = links.n_facets;
  if (!links.overflowed) {
    for (std::size_t x = 0; x < 3 * n; ++x)
      assert(links.across[x] >= 0);
    return;
  }
  links.records.allocate(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto &facet = links.facets[i];
    for (std::size_t e = 0; e < 3; ++e) {
      const Index a = facet[(e + 1) % 3], b = facet[(e + 2) % 3];
      links.records[3 * i + e] = {a < b ? a : b, a < b ? b : a,
                                  Index(3 * i + e)};
    }
  }
  std::sort(links.records.begin(), links.records.end(),
            [](const auto &x, const auto &y) {
              return x[0] < y[0] || (x[0] == y[0] && x[1] < y[1]);
            });
  for (std::size_t r = 0; r + 1 < links.records.size(); r += 2) {
    const auto &first = links.records[r], &second = links.records[r + 1];
    assert(first[0] == second[0] && first[1] == second[1]);
    links.across[std::size_t(first[2])] = second[2];
    links.across[std::size_t(second[2])] = first[2];
  }
}

} // namespace tf::topology::cdt::dt3
