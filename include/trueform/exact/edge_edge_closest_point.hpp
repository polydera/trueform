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

#include "./meta.hpp"
#include "./orient3d.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// Closest-approach contact point between segments (a0,a1) and (b0,b1),
/// placed on segment A. The rational parameter is exact and clamped to
/// the segment; the denominator is shift-reduced before the final
/// interpolation so the product stays in T2 (the sub-lattice loss is
/// irrelevant at contact scale). A parallel pair (zero denominator)
/// takes the midpoint of B's projected overlap on A. When the segments
/// truly cross, the closest approach IS the crossing.
template <typename Int>
auto edge_edge_closest_point(const pt3<Int> &a0, const pt3<Int> &a1,
                             const pt3<Int> &b0, const pt3<Int> &b1)
    -> pt3<Int> {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 u[3], v[3], w[3];
  for (int i = 0; i < 3; ++i) {
    u[i] = T1(a1[i]) - T1(a0[i]);
    v[i] = T1(b1[i]) - T1(b0[i]);
    w[i] = T1(a0[i]) - T1(b0[i]);
  }
  auto dot = [](const T1 *x, const T1 *y) -> T2 {
    return T2(x[0]) * T2(y[0]) + T2(x[1]) * T2(y[1]) + T2(x[2]) * T2(y[2]);
  };
  const T2 a = dot(u, u);
  const T2 b = dot(u, v);
  const T2 c = dot(v, v);
  const T2 d = dot(u, w);
  T2 den = a * c - b * b;
  T2 num = b * dot(v, w) - c * d;
  if (!(T2(0) < den)) {
    // parallel: midpoint of B's projected overlap on A, (b - 2d)/(2a)
    num = b - d - d;
    den = a + a;
    if (!(T2(0) < den))
      return a0;
  }
  if (num < T2(0))
    num = T2(0);
  if (den < num)
    num = den;
  // keep den * |u[i]| within T2
  constexpr unsigned k_lim = sizeof(Int) == 8 ? 180 : 90;
  const T2 lim = T2(1) << k_lim;
  while (lim < den) {
    den = den >> 1;
    num = num >> 1;
  }
  if (!(T2(0) < den))
    return a0;
  pt3<Int> point;
  for (int i = 0; i < 3; ++i)
    point[i] =
        a0[i] + static_cast<Int>(div_round(num * T2(u[i]), den));
  return point;
}

/// Closest point on segment (b0,b1) to `p`.
template <typename Int>
auto segment_closest_point(const pt3<Int> &b0, const pt3<Int> &b1,
                           const pt3<Int> &p) -> pt3<Int> {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 v[3], w[3];
  for (int i = 0; i < 3; ++i) {
    v[i] = T1(b1[i]) - T1(b0[i]);
    w[i] = T1(p[i]) - T1(b0[i]);
  }
  T2 den = T2(v[0]) * T2(v[0]) + T2(v[1]) * T2(v[1]) + T2(v[2]) * T2(v[2]);
  T2 num = T2(v[0]) * T2(w[0]) + T2(v[1]) * T2(w[1]) + T2(v[2]) * T2(w[2]);
  if (!(T2(0) < den))
    return b0;
  if (num < T2(0))
    num = T2(0);
  if (den < num)
    num = den;
  constexpr unsigned k_lim = sizeof(Int) == 8 ? 180 : 90;
  const T2 lim = T2(1) << k_lim;
  while (lim < den) {
    den = den >> 1;
    num = num >> 1;
  }
  if (!(T2(0) < den))
    return b0;
  pt3<Int> point;
  for (int i = 0; i < 3; ++i)
    point[i] = b0[i] + static_cast<Int>(div_round(num * T2(v[i]), den));
  return point;
}

} // namespace tf::exact
