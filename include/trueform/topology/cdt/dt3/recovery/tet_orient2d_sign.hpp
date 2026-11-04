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
#include "../../../../exact/float_filter_sound.hpp"
#include "../../../../exact/meta.hpp"
#include "../../../../exact/orient2d_filter_sign.hpp"
#include "../../../../exact/vertex.hpp"

namespace tf::topology::cdt::dt3 {

/// The orient2d sign of three points of the `Int` lattice carried in
/// `Coord`: @ref tf::exact::orient2d_filter_sign decides wherever its bound
/// does and @ref tf::exact::float_filter_sound holds, and the lattice's own
/// ladder decides the rest exactly.
template <typename Int, typename Coord>
auto tet_orient2d_sign(const tf::exact::pt2<Coord> &a,
                       const tf::exact::pt2<Coord> &b,
                       const tf::exact::pt2<Coord> &c) -> int {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  if constexpr (tf::exact::float_filter_sound) {
    const int filtered = tf::exact::orient2d_filter_sign<Int>(a, b, c);
    if (filtered)
      return filtered;
  }
  const T2 determinant = T2(T1(b[0]) - a[0]) * (T1(c[1]) - a[1]) -
                         T2(T1(b[1]) - a[1]) * (T1(c[0]) - a[0]);
  return determinant > T2(0) ? 1 : determinant < T2(0) ? -1 : 0;
}

} // namespace tf::topology::cdt::dt3
