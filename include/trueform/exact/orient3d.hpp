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
/// the fourth point is at the origin. Takes int64 to avoid int32 overflow
/// from the translation step.
inline auto orient3d_sos_presorted(const std::array<int64_t, 3> &a,
                                    const std::array<int64_t, 3> &b,
                                    const std::array<int64_t, 3> &c) -> bool {
  using I128 = int128;
  int64_t bx = b[0], by = b[1], bz = b[2];
  int64_t cx = c[0], cy = c[1], cz = c[2];
  I128 cross_x = I128(by) * cz - I128(bz) * cy;
  I128 cross_y = I128(bz) * cx - I128(bx) * cz;
  I128 cross_z = I128(bx) * cy - I128(by) * cx;
  I128 det =
      I128(a[0]) * cross_x + I128(a[1]) * cross_y + I128(a[2]) * cross_z;
  if (det)
    return det > 0;

  // SoS perturbation cascade
  int64_t v;
  v = bx * cy - by * cx;
  if (v)
    return v > 0;
  v = -(bx * cz - bz * cx);
  if (v)
    return v > 0;
  v = by * cz - bz * cy;
  if (v)
    return v > 0;

  int64_t ax = a[0], ay = a[1], az = a[2];
  v = -(ax * cy - ay * cx);
  if (v)
    return v > 0;
  if (cx)
    return cx > 0;
  if (cy)
    return cy < 0;

  v = ax * cz - az * cx;
  if (v)
    return v > 0;
  if (cz)
    return cz > 0;

  v = ax * by - ay * bx;
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

  std::array<int64_t, 3> a, b, c;
  for (int i = 0; i < 3; ++i) {
    a[i] = int64_t(pa[i]) - int64_t(pd[i]);
    b[i] = int64_t(pb[i]) - int64_t(pd[i]);
    c[i] = int64_t(pc[i]) - int64_t(pd[i]);
  }

  return odd != orient3d_sos_presorted(a, b, c);
}

inline auto orient3d_sos(const std::array<vertex, 4> &vs) -> bool {
  return orient3d_sos(vs.data());
}

/// Exact orient3d volume (signed). Used for barycentric weight computation.
inline auto orient3d_value(const pt3 &a, const pt3 &b, const pt3 &c,
                           const pt3 &d) -> int128 {
  int64_t ax = int64_t(b[0]) - a[0], ay = int64_t(b[1]) - a[1],
          az = int64_t(b[2]) - a[2];
  int64_t bx = int64_t(c[0]) - a[0], by = int64_t(c[1]) - a[1],
          bz = int64_t(c[2]) - a[2];
  int64_t cx = int64_t(d[0]) - a[0], cy = int64_t(d[1]) - a[1],
          cz = int64_t(d[2]) - a[2];
  using I = int128;
  return I(ax) * (I(by) * cz - I(bz) * cy) -
         I(ay) * (I(bx) * cz - I(bz) * cx) +
         I(az) * (I(bx) * cy - I(by) * cx);
}

/// Exact orient3d sign (no SoS). Returns -1, 0, or +1.
inline auto orient3d_sign(const std::array<vertex, 4> &vs) -> int {
  auto val = orient3d_value(vs[0].pt, vs[1].pt, vs[2].pt, vs[3].pt);
  return (val > 0) ? 1 : (val < 0) ? -1 : 0;
}

} // namespace tf::exact
