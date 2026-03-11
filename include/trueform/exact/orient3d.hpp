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
#include <array>

namespace tf::exact {

/// Integer rounding division (no floating-point cast).
inline auto div_round(int128 n, int128 d) -> int128 {
  return ((n < 0) == (d < 0)) ? ((n + d / 2) / d) : ((n - d / 2) / d);
}

/// Core SoS orient3d of (a, b, c) relative to origin.
/// Points must be pre-sorted by vertex ID and translated so that
/// the fourth point is at the origin.
inline auto orient3d_sos_presorted(const pt3 &a, const pt3 &b, const pt3 &c) -> bool {
  using I128 = int128;
  using I64 = int64_t;

  I64 bx = b[0], by = b[1], bz = b[2];
  I64 cx = c[0], cy = c[1], cz = c[2];
  I128 cross_x = I128(by) * cz - I128(bz) * cy;
  I128 cross_y = I128(bz) * cx - I128(bx) * cz;
  I128 cross_z = I128(bx) * cy - I128(by) * cx;
  I128 det =
      I128(a[0]) * cross_x + I128(a[1]) * cross_y + I128(a[2]) * cross_z;
  if (det)
    return det > 0;

  // SoS perturbation cascade
  I64 v;
  v = bx * cy - by * cx;
  if (v)
    return v > 0;
  v = -(bx * cz - bz * cx);
  if (v)
    return v > 0;
  v = by * cz - bz * cy;
  if (v)
    return v > 0;

  I64 ax = a[0], ay = a[1], az = a[2];
  v = -(ax * cy - ay * cx);
  if (v)
    return v > 0;
  if (c[0])
    return c[0] > 0;
  if (c[1])
    return c[1] < 0;

  v = ax * cz - az * cx;
  if (v)
    return v > 0;
  if (c[2])
    return c[2] > 0;

  v = ax * by - ay * bx;
  if (v)
    return v > 0;
  if (b[0])
    return b[0] < 0;
  if (b[1])
    return b[1] > 0;
  if (a[0])
    return a[0] > 0;

  return true;
}

/// SoS orient3d for 4 vertices. Sorts by ID and applies parity correction.
/// Returns true if the fourth point is on the positive side of the
/// oriented plane defined by the first three.
inline auto orient3d_sos(const vertex *vs) -> bool {
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

  pt3 a, b, c;
  for (int i = 0; i < 3; ++i) {
    a[i] = pa[i] - pd[i];
    b[i] = pb[i] - pd[i];
    c[i] = pc[i] - pd[i];
  }

  return odd != orient3d_sos_presorted(a, b, c);
}

inline auto orient3d_sos(const std::array<vertex, 4> &vs) -> bool {
  return orient3d_sos(vs.data());
}

/// Exact orient3d volume (signed). Used for barycentric weight computation.
inline auto orient3d_value(const pt3 &a, const pt3 &b, const pt3 &c,
                           const pt3 &d) -> int128 {
  using I = int128;
  I ax = b[0] - a[0], ay = b[1] - a[1], az = b[2] - a[2];
  I bx = c[0] - a[0], by = c[1] - a[1], bz = c[2] - a[2];
  I cx = d[0] - a[0], cy = d[1] - a[1], cz = d[2] - a[2];
  return ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) +
         az * (bx * cy - by * cx);
}

/// Exact orient3d sign (no SoS). Returns -1, 0, or +1.
inline auto orient3d_sign(const std::array<vertex, 4> &vs) -> int {
  auto val = orient3d_value(vs[0].pt, vs[1].pt, vs[2].pt, vs[3].pt);
  return (val > 0) ? 1 : (val < 0) ? -1 : 0;
}

} // namespace tf::exact
