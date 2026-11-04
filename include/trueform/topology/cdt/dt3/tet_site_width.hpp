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
#include "../../../exact/insphere.hpp"
#include "../../../exact/meta.hpp"
#include <algorithm>
#include <limits>

namespace tf::topology::cdt::dt3 {

/// The bits a site of the `Int` lattice stored in `Coord` may span: what the
/// predicates on `Int` admit past its own coordinate bits, and no more than
/// `Coord` holds — 40 for int32 in int64, 63 for int64 in int64, 83 for
/// int64 in int128. `storage_bits` is `Coord`'s own digits, its bits less
/// the sign where its limits state none.
template <typename Int, typename Coord> struct tet_site_width {
  static constexpr int storage_bits = std::numeric_limits<Coord>::is_specialized
                                          ? std::numeric_limits<Coord>::digits
                                          : int(8 * sizeof(Coord)) - 1;
  static constexpr int value =
      std::min(tf::exact::meta<Int>::coordinate_bits +
                   tf::exact::insphere_scale_bits<Int>(),
               storage_bits);
};

} // namespace tf::topology::cdt::dt3
