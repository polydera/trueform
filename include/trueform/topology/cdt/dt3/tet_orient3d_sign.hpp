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
#include "../../../exact/float_filter_sound.hpp"
#include "../../../exact/orient3d.hpp"
#include "../../../exact/orient3d_filter_sign.hpp"
#include "../../../exact/vertex.hpp"
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The orient3d sign of four points of the `Int` lattice carried in
/// `Coord`: -1, 0 or +1. Past 32 bits the sign is read off the double images
/// of the differences first (@ref tf::exact::orient3d_filter_sign), the
/// exact value deciding only what the filter cannot; on int32 the exact value
/// costs what the filter costs, and a zero, common on a coplanar surface,
/// would pay both.
template <typename Int, typename Coord>
auto tet_orient3d_sign(const tf::exact::pt3<Coord> &a,
                       const tf::exact::pt3<Coord> &b,
                       const tf::exact::pt3<Coord> &c,
                       const tf::exact::pt3<Coord> &d) -> int {
  if constexpr (tf::exact::float_filter_sound &&
                sizeof(Int) > sizeof(std::int32_t)) {
    const int filtered = tf::exact::orient3d_filter_sign<Int>(a, b, c, d);
    if (filtered)
      return filtered;
  }
  const auto value = tf::exact::orient3d_value_scaled<Int>(a, b, c, d);
  return value > 0 ? 1 : value < 0 ? -1 : 0;
}

} // namespace tf::topology::cdt::dt3
