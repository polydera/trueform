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
#include "./hole_table.hpp"
#include <cstddef>

namespace tf::fill {

/// Gather the retained facets and their readings to the front, in the order
/// they were gathered in, which is what makes a facet's slot the name its two
/// states are counted from.
template <typename Index>
auto compact_hole_table_facets(tf::fill::hole_table<Index> &table) -> void {
  std::size_t kept = 0;
  for (std::size_t f = 0; f < table.retained.size(); ++f) {
    if (!table.retained[f])
      continue;
    if (kept != f) {
      table.facets[kept] = table.facets[f];
      table.normal[kept] = table.normal[f];
      table.area[kept] = table.area[f];
      table.boundary[kept] = table.boundary[f];
    }
    ++kept;
  }
  table.facets.erase_till_end(table.facets.begin() + std::ptrdiff_t(kept));
  table.normal.erase_till_end(table.normal.begin() + std::ptrdiff_t(kept));
  table.area.erase_till_end(table.area.begin() + std::ptrdiff_t(kept));
  table.boundary.erase_till_end(table.boundary.begin() + std::ptrdiff_t(kept));
}

} // namespace tf::fill
