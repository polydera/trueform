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
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Empties the table slots of the first `n_facets` facets' edges, read through
/// the local ids the open boundary gave their corners.
template <typename Index>
auto scrub_tet_cavity_table(tet_cavity_links<Index> &links,
                            std::size_t n_facets) -> void {
  constexpr std::size_t max_local = tet_cavity_links<Index>::max_local;
  for (std::size_t i = 0; i < n_facets; ++i) {
    const auto &facet = links.facets[i];
    std::uint32_t local[3];
    for (std::size_t q = 0; q < 3; ++q)
      local[q] = links.local_of_site[tet_cavity_stamp_slot(facet[q])];
    for (std::size_t e = 0; e < 3; ++e) {
      const std::uint32_t a = local[(e + 1) % 3], b = local[(e + 2) % 3];
      const std::uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
      links.table[std::size_t(lo) * max_local + std::size_t(hi)] = Index(-1);
    }
  }
}

} // namespace tf::topology::cdt::dt3
