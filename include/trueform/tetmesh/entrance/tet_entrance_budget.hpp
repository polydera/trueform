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
#include "../../topology/cdt/dt3/tet_site_width.hpp"
#include <algorithm>
#include <limits>

namespace tf::tetmesh {

/// How the entrance spends a site's bits. A site of the `Int` lattice
/// stored in `Coord` spans the `width` the predicates admit
/// (@ref tf::topology::cdt::dt3::tet_site_width): 40 for int32 stored in
/// int64, 63 for int64 in int64. A site spends it on the lattice the input
/// converts onto and on the frame a constraint's split reaches below it:
///
///   lattice_bits = min(converter_bits, width - frame_bits)
///
/// The frame is ten bits, and the lattice takes the rest up to the house
/// converter's own width for `Int`, the power of two below 0.99 of its
/// range: 30 of int32, 53 of int64 stored in int64. A point the recovery
/// needs finer than the frame is a site below it, as deep as the deep
/// lattice admits (@ref tf::topology::cdt::dt3::tet_deep_lattice): 23 bits
/// for int32 sites, 20 for int64.
template <typename Int, typename Coord> struct tet_entrance_budget {
  static constexpr int storage_bits =
      tf::topology::cdt::dt3::tet_site_width<Int, Coord>::storage_bits;
  static constexpr int width =
      tf::topology::cdt::dt3::tet_site_width<Int, Coord>::value;
  static constexpr int frame_bits = 10;
  static constexpr int converter_bits = std::numeric_limits<Int>::digits - 1;
  static constexpr int lattice_bits =
      std::min(converter_bits, width - frame_bits);
};

} // namespace tf::tetmesh
