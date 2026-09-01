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
#include "./vertex.hpp"
#include <utility>

namespace tf::exact {

/// The exact parameter of segment (p0, p1)'s crossing with segment
/// (q0, q1), both in one face plane, through that plane's projection
/// axes (@ref tf::exact::projection_axes). Projection along the
/// dominant axis is a bijection on the plane, so the projected
/// crossing is the crossing. Caller ensures the segments are not
/// parallel in projection.
template <typename Int>
auto make_edge_edge_parameter(const pt3<Int> &p0, const pt3<Int> &p1,
                              const pt3<Int> &q0, const pt3<Int> &q1,
                              const std::pair<int, int> &axes)
    -> edge_parameter<Int> {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  const int ax = axes.first;
  const int ay = axes.second;
  const T1 ux = T1(p1[ax]) - p0[ax], uy = T1(p1[ay]) - p0[ay];
  const T1 vx = T1(q1[ax]) - q0[ax], vy = T1(q1[ay]) - q0[ay];
  const T1 wx = T1(q0[ax]) - p0[ax], wy = T1(q0[ay]) - p0[ay];
  T2 num = T2(wx) * vy - T2(wy) * vx;
  T2 den = T2(ux) * vy - T2(uy) * vx;
  if (den < T2(0)) {
    num = -num;
    den = -den;
  }
  return {num, den};
}

} // namespace tf::exact
