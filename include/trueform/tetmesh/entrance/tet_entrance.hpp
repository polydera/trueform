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
#include "./tet_entrance_surface.hpp"

namespace tf::tetmesh {

/// What the entrance hands the mesher: the `surface` read on the `Int`
/// lattice, with the converter that maps it back, whose points the mesher
/// mints at `2^frame_bits` of the lattice in `Coord`.
template <typename Index, typename RealT, typename Int, typename Coord>
struct tet_entrance {
  using int_type = Int;
  using coord_type = Coord;
  tet_entrance_surface<Index, RealT, Int> surface;
  int frame_bits = 0;
};

} // namespace tf::tetmesh
