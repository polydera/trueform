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
#include "../../exact/pt_converter.hpp"
#include "./tet_entrance_surface.hpp"

namespace tf::tetmesh {

/// What the entrance hands the mesher: the `surface` read on the `Int`
/// lattice, whose points the mesher mints at `2^frame_bits` of the lattice
/// in `Coord`, with its converter.
template <typename Index, typename RealT, typename Int, typename Coord>
struct tet_entrance {
  using int_type = Int;
  using coord_type = Coord;
  tet_entrance_surface<Index, RealT, Int> surface;
  tf::exact::pt_converter<Int, RealT, 3> converter;
  int frame_bits = 0;
};

} // namespace tf::tetmesh
