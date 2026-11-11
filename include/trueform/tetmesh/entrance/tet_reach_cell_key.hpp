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
#pragma once
#include <array>

namespace tf::tetmesh {

/// A point's reach cell beside the point, ordered by the cell and then the
/// point, so a sort groups the points of one cell with its smallest first.
template <typename Index, typename Int> struct tet_reach_cell_key {
  std::array<Int, 3> cell;
  Index point;

  friend auto operator<(const tet_reach_cell_key &a,
                        const tet_reach_cell_key &b) -> bool {
    return a.cell < b.cell || (a.cell == b.cell && a.point < b.point);
  }
};

} // namespace tf::tetmesh
