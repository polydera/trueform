/*
 * Copyright (c) 2025 XLAB
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
#include "./vertex.hpp"

namespace tf::exact {

/// Exact orient2d: positive if c is left of a->b, negative if right, zero if
/// collinear. Exact — products use meta<Int>::T2 to avoid overflow.
template <typename Int>
auto orient2d(const pt2<Int> &a, const pt2<Int> &b, const pt2<Int> &c)
    -> typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  return T2(T1(b[0]) - T1(a[0])) * T2(T1(c[1]) - T1(a[1])) -
         T2(T1(b[1]) - T1(a[1])) * T2(T1(c[0]) - T1(a[0]));
}

/// Exact squared distance between two 2D points.
template <typename Int>
auto distance2(const pt2<Int> &a, const pt2<Int> &b) ->
    typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 dx = T1(a[0]) - T1(b[0]);
  T1 dy = T1(a[1]) - T1(b[1]);
  return T2(dx) * dx + T2(dy) * dy;
}

/// Exact orient2d sign for 2D points. Direct wide-integer evaluation:
/// measured faster than a floating-point filter on both int32 and int64
/// lattices.
template <typename Int>
auto orient2d_sign(const pt2<Int> &a, const pt2<Int> &b, const pt2<Int> &c)
    -> int {
  auto det = orient2d(a, b, c);
  return (det > 0) ? 1 : (det < 0) ? -1 : 0;
}

/// Exact orient2d sign for 3 vertices projected to 2D via axes (ax0, ax1).
/// Returns -1, 0, or +1.
template <typename Index, typename Int>
auto orient2d_sign(const vertex<Index, Int> &v0, const vertex<Index, Int> &v1,
                   const vertex<Index, Int> &v2, int ax0, int ax1) -> int {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T2 det = T2(T1(v1.pt[ax0]) - T1(v0.pt[ax0])) *
               T2(T1(v2.pt[ax1]) - T1(v0.pt[ax1])) -
           T2(T1(v1.pt[ax1]) - T1(v0.pt[ax1])) *
               T2(T1(v2.pt[ax0]) - T1(v0.pt[ax0]));
  return (det > 0) ? 1 : (det < 0) ? -1 : 0;
}

/// SoS orient2d for 3 vertices projected to 2D via axes (ax0, ax1).
/// Sort by ID, translate to last as origin, compute det, SoS cascade if 0.
/// Returns true for positive orientation (after parity correction).
template <typename Index, typename Int>
auto orient2d_sos(const vertex<Index, Int> &v0, const vertex<Index, Int> &v1,
                  const vertex<Index, Int> &v2, int ax0, int ax1) -> bool {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  std::array<int, 3> order = {0, 1, 2};
  const vertex<Index, Int> *vs[3] = {&v0, &v1, &v2};
  bool odd = false;
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (vs[order[i]]->id > vs[order[j]]->id) {
        odd = !odd;
        std::swap(order[i], order[j]);
      }
  T1 a0 = T1(vs[order[0]]->pt[ax0]) - T1(vs[order[2]]->pt[ax0]);
  T1 a1 = T1(vs[order[0]]->pt[ax1]) - T1(vs[order[2]]->pt[ax1]);
  T1 b0 = T1(vs[order[1]]->pt[ax0]) - T1(vs[order[2]]->pt[ax0]);
  T1 b1 = T1(vs[order[1]]->pt[ax1]) - T1(vs[order[2]]->pt[ax1]);
  T2 det = T2(a0) * b1 - T2(a1) * b0;
  if (det)
    return (det > 0) != odd;
  // SoS cascade (cofactors of 3x3 homogeneous orient2d matrix)
  if (b1)
    return (b1 > 0) != odd;
  if (b0)
    return (b0 < 0) != odd;
  if (a1)
    return (a1 < 0) != odd;
  if (a0)
    return (a0 > 0) != odd;
  T1 d1 = a1 - b1;
  if (d1)
    return (d1 > 0) != odd;
  T1 d0 = b0 - a0;
  if (d0)
    return (d0 > 0) != odd;
  return !odd;
}

} // namespace tf::exact
