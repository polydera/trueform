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
#include "./edge_parameter.hpp"
#include "./meta.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// Orthogonal projection onto the line (a,b), as an exact unrounded fraction.
/// Coordinates must fit the differences and degree-two products of Int.
template <typename Int, typename Coord>
auto make_edge_projection_parameter(const pt3<Coord> &a, const pt3<Coord> &b,
                                    const pt3<Coord> &q)
    -> edge_parameter<Int> {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  const T1 dx = T1(b[0]) - a[0], dy = T1(b[1]) - a[1], dz = T1(b[2]) - a[2];
  const T1 wx = T1(q[0]) - a[0], wy = T1(q[1]) - a[1], wz = T1(q[2]) - a[2];
  return {T2(wx) * dx + T2(wy) * dy + T2(wz) * dz,
          T2(dx) * dx + T2(dy) * dy + T2(dz) * dz};
}

} // namespace tf::exact
