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
#include "../../core/polygons.hpp"
#include "../../exact/pt_converter.hpp"
#include "./make_tet_entrance_surface.hpp"
#include "./tet_entrance.hpp"
#include "./tet_entrance_budget.hpp"
#include "./tet_entrance_surface.hpp"
#include <utility>

namespace tf::tetmesh {

/// Bring `surface` to the mesher on the `Int` lattice stored in `Coord`:
/// every point stands at `2^frame_bits` of the lattice, the depth
/// `tet_entrance_budget` leaves the split frames, and the surface moves into
/// the entrance.
template <typename Index, typename Int, typename Coord, typename RealT>
auto make_tet_entrance(tet_entrance_surface<Index, RealT, Int> surface) {
  tet_entrance<Index, RealT, Int, Coord> entrance;
  entrance.frame_bits =
      tet_entrance_budget<Int, Coord>::width - surface.input_bits;
  entrance.converter = tf::exact::pt_converter<Int, RealT, 3>{
      {surface.converter.scale, surface.converter.offset}};
  entrance.surface = std::move(surface);
  return entrance;
}

/// Bring `polygons` to the mesher on the `Int` lattice stored in `Coord`,
/// its surface (@ref make_tet_entrance_surface) read on that lattice.
template <typename Index, typename Int, typename Coord, typename Policy>
auto make_tet_entrance(const tf::polygons<Policy> &polygons) {
  return make_tet_entrance<Index, Int, Coord>(
      make_tet_entrance_surface<Index, Int, Coord>(polygons));
}

} // namespace tf::tetmesh
