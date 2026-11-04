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
#include "./tet_site_position.hpp"
#include "./tetrahedralization_owner.hpp"
#include <algorithm>

namespace tf::topology::cdt::dt3 {

/// The deepest any of the sites `slots` of `lanes` stands below the frame,
/// each read off its deep row.
template <typename Index, typename Coord, typename... Slots>
auto tet_deepest_row_of(const tet_site_lanes<Index, Coord> &lanes,
                        Slots... slots) -> int {
  int deepest = 0;
  ((deepest = std::max(deepest, lanes.depth(Index(slots)))), ...);
  return deepest;
}

/// The deepest any of the sites `slots` stands below the frame, a
/// decision's depth over them; zero at once while every site stands on it.
template <typename Index, typename Coord, typename Int, typename... Slots>
auto tet_deepest_of(const tetrahedralization_owner<Index, Coord, Int> &owner,
                    Slots... slots) -> int {
  const auto lanes = tet_site_lanes_of(owner);
  return lanes.deepest ? tet_deepest_row_of(lanes, slots...) : 0;
}

} // namespace tf::topology::cdt::dt3
