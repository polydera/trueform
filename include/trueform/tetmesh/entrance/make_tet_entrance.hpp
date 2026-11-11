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
#include "../../core/coordinate_type.hpp"
#include "../../core/polygons.hpp"
#include "./make_tet_entrance_surface.hpp"
#include "./tet_entrance.hpp"
#include "./tet_entrance_budget.hpp"

namespace tf::tetmesh {

/// Bring `polygons` to the mesher on the `Int` lattice stored in `Coord`:
/// its surface (@ref make_tet_entrance_surface) read on that lattice, as
/// given and placed by the door under `tolerance`, every point standing at
/// `2^frame_bits` of the lattice, the depth `tet_entrance_budget` leaves the
/// split frames.
template <typename Index, typename Int, typename Coord, typename Policy>
auto make_tet_entrance(const tf::polygons<Policy> &polygons, double tolerance) {
  tet_entrance<Index, tf::coordinate_type<Policy>, Int, Coord> entrance{
      make_tet_entrance_surface<Index, Int, Coord>(polygons, tolerance)};
  entrance.frame_bits =
      tet_entrance_budget<Int, Coord>::width - entrance.surface.input_bits;
  return entrance;
}

} // namespace tf::tetmesh
