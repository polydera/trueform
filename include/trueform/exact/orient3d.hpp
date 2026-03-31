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
#include <array>

namespace tf::exact {

/// Integer rounding division (no floating-point cast).
template <typename T> auto div_round(T n, T d) -> T {
  return ((n < 0) == (d < 0)) ? ((n + d / 2) / d) : ((n - d / 2) / d);
}

/// Core SoS orient3d of (a, b, c) relative to origin.
/// Points must be pre-sorted by vertex ID and translated so that
/// the fourth point is at the origin. Takes T1 to avoid T0 overflow
/// from the translation step.
template <typename Int>
auto orient3d_sos_presorted(const std::array<typename meta<Int>::T1, 3> &a,
                            const std::array<typename meta<Int>::T1, 3> &b,
                            const std::array<typename meta<Int>::T1, 3> &c)
    -> bool {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 bx = b[0], by = b[1], bz = b[2];
  T1 cx = c[0], cy = c[1], cz = c[2];
  T2 cross_x = T2(by) * cz - T2(bz) * cy;
  T2 cross_y = T2(bz) * cx - T2(bx) * cz;
  T2 cross_z = T2(bx) * cy - T2(by) * cx;
  T2 det = T2(a[0]) * cross_x + T2(a[1]) * cross_y + T2(a[2]) * cross_z;
  if (det)
    return det > 0;

  // SoS perturbation cascade
  T2 v;
  v = T2(bx) * cy - T2(by) * cx;
  if (v)
    return v > 0;
  v = -(T2(bx) * cz - T2(bz) * cx);
  if (v)
    return v > 0;
  v = T2(by) * cz - T2(bz) * cy;
  if (v)
    return v > 0;

  T1 ax = a[0], ay = a[1], az = a[2];
  v = -(T2(ax) * cy - T2(ay) * cx);
  if (v)
    return v > 0;
  if (cx)
    return cx > 0;
  if (cy)
    return cy < 0;

  v = T2(ax) * cz - T2(az) * cx;
  if (v)
    return v > 0;
  if (cz)
    return cz > 0;

  v = T2(ax) * by - T2(ay) * bx;
  if (v)
    return v > 0;
  if (bx)
    return bx < 0;
  if (by)
    return by > 0;
  if (a[0])
    return a[0] > 0;

  return true;
}

/// SoS orient3d for 4 vertices. Sorts by ID and applies parity correction.
/// Returns true if the fourth point is on the positive side of the
/// oriented plane defined by the first three.
template <typename Index, typename Int>
auto orient3d_sos(const vertex<Index, Int> *vs) -> bool {
  using T1 = typename meta<Int>::T1;
  bool odd = false;
  std::array<int, 4> order = {0, 1, 2, 3};

  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      if (vs[order[i]].id > vs[order[j]].id) {
        odd = !odd;
        std::swap(order[i], order[j]);
      }
    }
  }

  auto &pa = vs[order[0]].pt;
  auto &pb = vs[order[1]].pt;
  auto &pc = vs[order[2]].pt;
  auto &pd = vs[order[3]].pt;

  std::array<T1, 3> a, b, c;
  for (int i = 0; i < 3; ++i) {
    a[i] = T1(pa[i]) - T1(pd[i]);
    b[i] = T1(pb[i]) - T1(pd[i]);
    c[i] = T1(pc[i]) - T1(pd[i]);
  }

  return odd != orient3d_sos_presorted<Int>(a, b, c);
}

template <typename Index, typename Int>
auto orient3d_sos(const std::array<vertex<Index, Int>, 4> &vs) -> bool {
  return orient3d_sos(vs.data());
}

/// Exact orient3d volume (signed). Used for barycentric weight computation.
template <typename Int>
auto orient3d_value(const pt3<Int> &a, const pt3<Int> &b, const pt3<Int> &c,
                    const pt3<Int> &d) -> typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 ax = T1(b[0]) - a[0], ay = T1(b[1]) - a[1], az = T1(b[2]) - a[2];
  T1 bx = T1(c[0]) - a[0], by = T1(c[1]) - a[1], bz = T1(c[2]) - a[2];
  T1 cx = T1(d[0]) - a[0], cy = T1(d[1]) - a[1], cz = T1(d[2]) - a[2];
  return T2(ax) * (T2(by) * cz - T2(bz) * cy) -
         T2(ay) * (T2(bx) * cz - T2(bz) * cx) +
         T2(az) * (T2(bx) * cy - T2(by) * cx);
}

/// Exact orient3d sign (no SoS). Returns -1, 0, or +1.
template <typename Index, typename Int>
auto orient3d_sign(const std::array<vertex<Index, Int>, 4> &vs) -> int {
  auto val = orient3d_value(vs[0].pt, vs[1].pt, vs[2].pt, vs[3].pt);
  return (val > 0) ? 1 : (val < 0) ? -1 : 0;
}

} // namespace tf::exact
