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
#include "../../core/small_vector.hpp"
#include "./make_plane_radial_fans.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Whether some domain stands an odd number of times around ring
///        `fan`: each page names its component by its first occurrence and
///        contributes both of that component's sides (`domain_of_side`).
///
/// Around a ring whose order is stated every sector consumes two facing
/// sides, so every domain occurs an even number of times; an odd domain is
/// a merge only the ring's order can state.
template <typename Index, typename Labels, typename ExposedOfRow>
auto ring_domain_parity_is_odd(const Labels &labels,
                               const plane_radial_fans<Index> &fans,
                               const ExposedOfRow &exposed_of_row,
                               const tf::buffer<Index> &domain_of_side,
                               std::size_t fan) -> bool {
  tf::small_vector<Index, 16> domains;
  for (auto p = fans.page_offsets[fan]; p < fans.page_offsets[fan + 1]; ++p) {
    const auto rows = fans.rows[std::size_t(p)];
    if (!rows.size())
      continue;
    const Index row = rows[0];
    Index component;
    if (row < Index(0))
      component = Index(-row - 1);
    else {
      const Index exposed = exposed_of_row[std::size_t(row / Index(3))];
      if (exposed == Index(-1))
        continue;
      component = labels[exposed];
    }
    if (component < Index(0))
      continue;
    domains.push_back(domain_of_side[2 * std::size_t(component)]);
    domains.push_back(domain_of_side[2 * std::size_t(component) + 1]);
  }
  std::sort(domains.begin(), domains.end());
  for (std::size_t i = 0; i < domains.size();) {
    std::size_t j = i;
    while (j < domains.size() && domains[j] == domains[i])
      ++j;
    if ((j - i) % 2)
      return true;
    i = j;
  }
  return false;
}

} // namespace tf::csg::graph
