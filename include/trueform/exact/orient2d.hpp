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

#include "./int128.hpp"
#include "./vertex.hpp"
#include <cstdint>

namespace tf::exact {

/// Exact orient2d: positive if c is left of a->b, negative if right, zero if
/// collinear. Exact for int32 inputs — products use int128 to avoid overflow.
inline auto orient2d(const pt2 &a, const pt2 &b, const pt2 &c) -> int128 {
  return int128(int64_t(b[0]) - int64_t(a[0])) *
             int128(int64_t(c[1]) - int64_t(a[1])) -
         int128(int64_t(b[1]) - int64_t(a[1])) *
             int128(int64_t(c[0]) - int64_t(a[0]));
}

/// Exact squared distance between two 2D int32 points.
inline auto distance2(const pt2 &a, const pt2 &b) -> int128 {
  int64_t dx = int64_t(a[0]) - int64_t(b[0]),
          dy = int64_t(a[1]) - int64_t(b[1]);
  return int128(dx) * dx + int128(dy) * dy;
}

/// Exact orient2d sign for 3 vertices projected to 2D via axes (ax0, ax1).
/// Returns -1, 0, or +1.
template <typename Index>
auto orient2d_sign(const vertex<Index> &v0, const vertex<Index> &v1,
                   const vertex<Index> &v2, int ax0, int ax1) -> int {
  int128 det = int128(int64_t(v1.pt[ax0]) - int64_t(v0.pt[ax0])) *
                   int128(int64_t(v2.pt[ax1]) - int64_t(v0.pt[ax1])) -
               int128(int64_t(v1.pt[ax1]) - int64_t(v0.pt[ax1])) *
                   int128(int64_t(v2.pt[ax0]) - int64_t(v0.pt[ax0]));
  return (det > 0) ? 1 : (det < 0) ? -1 : 0;
}

/// SoS orient2d for 3 vertices projected to 2D via axes (ax0, ax1).
/// Sort by ID, translate to last as origin, compute det, SoS cascade if 0.
/// Returns true for positive orientation (after parity correction).
template <typename Index>
auto orient2d_sos(const vertex<Index> &v0, const vertex<Index> &v1,
                  const vertex<Index> &v2, int ax0, int ax1) -> bool {
  std::array<int, 3> order = {0, 1, 2};
  const vertex<Index> *vs[3] = {&v0, &v1, &v2};
  bool odd = false;
  for (int i = 0; i < 2; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (vs[order[i]]->id > vs[order[j]]->id) {
        odd = !odd;
        std::swap(order[i], order[j]);
      }
  int64_t a0 = int64_t(vs[order[0]]->pt[ax0]) - int64_t(vs[order[2]]->pt[ax0]);
  int64_t a1 = int64_t(vs[order[0]]->pt[ax1]) - int64_t(vs[order[2]]->pt[ax1]);
  int64_t b0 = int64_t(vs[order[1]]->pt[ax0]) - int64_t(vs[order[2]]->pt[ax0]);
  int64_t b1 = int64_t(vs[order[1]]->pt[ax1]) - int64_t(vs[order[2]]->pt[ax1]);
  int128 det = int128(a0) * b1 - int128(a1) * b0;
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
  int64_t d1 = a1 - b1;
  if (d1)
    return (d1 > 0) != odd;
  int64_t d0 = b0 - a0;
  if (d0)
    return (d0 > 0) != odd;
  return !odd;
}

} // namespace tf::exact
