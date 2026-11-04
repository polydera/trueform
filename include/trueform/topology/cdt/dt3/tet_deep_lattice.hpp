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
#include "../../../exact/int128.hpp"
#include "../../../exact/int32.hpp"
#include "../../../exact/int64.hpp"
#include "./tet_site_width.hpp"
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Where a decision reads the sites of the `Int` lattice stored in `Coord`
/// once one of its corners stands below the frame: the lattice `int_type`
/// its values run on and the `coord_type` an aligned site is read in. `cap`
/// is the depth a site may stand below the frame, the width that lattice
/// admits past the frame's own (@ref tet_site_width): 23 for int32 sites in
/// int64, read on int64; 20 for int64 sites in int64, read on int64 in
/// int128. A lattice with no such pair holds no site below its frame.
template <typename Int, typename Coord> struct tet_deep_lattice {
  using int_type = Int;
  using coord_type = Coord;
  static constexpr int cap = 0;
};

template <> struct tet_deep_lattice<tf::exact::int32, std::int64_t> {
  using int_type = tf::exact::int64;
  using coord_type = std::int64_t;
  static constexpr int cap =
      tet_site_width<int_type, coord_type>::value -
      tet_site_width<tf::exact::int32, std::int64_t>::value;
};

template <> struct tet_deep_lattice<tf::exact::int64, std::int64_t> {
  using int_type = tf::exact::int64;
  using coord_type = tf::exact::int128;
  static constexpr int cap =
      tet_site_width<int_type, coord_type>::value -
      tet_site_width<tf::exact::int64, std::int64_t>::value;
};

} // namespace tf::topology::cdt::dt3
