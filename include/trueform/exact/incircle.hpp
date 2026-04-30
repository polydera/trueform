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

/// Exact in_circle: positive if d is inside circumcircle of (a, b, c) when CCW.
/// Negative if outside, zero if on circle.
/// Determinant of:
/// | ax ay ax^2+ay^2 1 |
/// | bx by bx^2+by^2 1 |
/// | cx cy cx^2+cy^2 1 |
/// | dx dy dx^2+dy^2 1 |
/// Translated to d as origin:
/// | adx ady adx^2+ady^2 |
/// | bdx bdy bdx^2+bdy^2 |
/// | cdx cdy cdx^2+cdy^2 |
template <typename Int>
auto incircle(const pt2<Int> &a, const pt2<Int> &b, const pt2<Int> &c,
              const pt2<Int> &d) -> typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;

  T1 adx = T1(a[0]) - T1(d[0]);
  T1 ady = T1(a[1]) - T1(d[1]);
  T1 bdx = T1(b[0]) - T1(d[0]);
  T1 bdy = T1(b[1]) - T1(d[1]);
  T1 cdx = T1(c[0]) - T1(d[0]);
  T1 cdy = T1(c[1]) - T1(d[1]);

  T2 alift = T2(adx) * adx + T2(ady) * ady;
  T2 blift = T2(bdx) * bdx + T2(bdy) * bdy;
  T2 clift = T2(cdx) * cdx + T2(cdy) * cdy;

  return alift * (T2(bdx) * cdy - T2(bdy) * cdx) +
         blift * (T2(cdx) * ady - T2(cdy) * adx) +
         clift * (T2(adx) * bdy - T2(ady) * bdx);
}

/// Exact incircle sign. Returns -1 (outside), 0 (on), +1 (inside).
/// Assumes (a, b, c) is CCW.
template <typename Int>
auto incircle_sign(const pt2<Int> &a, const pt2<Int> &b, const pt2<Int> &c,
                   const pt2<Int> &d) -> int {
  auto val = incircle(a, b, c, d);
  return (val > 0) ? 1 : (val < 0) ? -1 : 0;
}
} // namespace tf::exact
