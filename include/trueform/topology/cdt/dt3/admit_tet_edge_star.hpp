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
#include "./tet_edge_star.hpp"
#include <algorithm>

namespace tf::topology::cdt::dt3 {

/// Append `cell` to `star` unless the star holds it already: true the first
/// time.
template <typename Index>
auto admit_tet_edge_star(tet_edge_star<Index> &star, Index cell) -> bool {
  auto &cells = star.cells;
  auto &hashed = star.hashed;
  if (cells.size() < 48) {
    if (std::find(cells.begin(), cells.end(), cell) != cells.end())
      return false;
  } else if (!hashed.insert(cell).second)
    return false;
  cells.push_back(cell);
  if (cells.size() == 48) {
    hashed.clear();
    hashed.insert(cells.begin(), cells.end());
  }
  return true;
}

} // namespace tf::topology::cdt::dt3
