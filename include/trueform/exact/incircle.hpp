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

/// Exact incircle sign. Returns -1 (outside), 0 (on), +1 (inside);
/// assumes (a, b, c) is CCW. int32 coordinates with all diffs below
/// 2^30 evaluate lifts and minors in T1 with three T2 products -- one
/// exact evaluation, no filter; otherwise the exact determinant runs
/// directly (measured faster than a floating-point filter). int64
/// coordinates run a Shewchuk semi-static double filter first -- T2 is
/// int256 there and the filter wins decisively on local operands --
/// falling back to the exact determinant only when it cannot decide.
/// The sign is exact in every path.
template <typename Int>
auto incircle_sign(const pt2<Int> &a, const pt2<Int> &b, const pt2<Int> &c,
                   const pt2<Int> &d) -> int {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 adxi = T1(a[0]) - T1(d[0]), adyi = T1(a[1]) - T1(d[1]);
  T1 bdxi = T1(b[0]) - T1(d[0]), bdyi = T1(b[1]) - T1(d[1]);
  T1 cdxi = T1(c[0]) - T1(d[0]), cdyi = T1(c[1]) - T1(d[1]);

  if constexpr (std::is_same_v<Int, int32>) {
    // lifts and 2x2 minors fit T1 (<= 2^61); each lift*minor fits T2
    // (sum 3*2^122 < 2^127); neighbouring triangulation vertices are
    // local, so this covers virtually every call
    const T1 lim30 = T1(1) << 30;
    auto in30 = [&](T1 v) { return v > -lim30 && v < lim30; };
    if (in30(adxi) && in30(adyi) && in30(bdxi) && in30(bdyi) &&
        in30(cdxi) && in30(cdyi)) {
      T1 alift = adxi * adxi + adyi * adyi;
      T1 blift = bdxi * bdxi + bdyi * bdyi;
      T1 clift = cdxi * cdxi + cdyi * cdyi;
      T1 m_a = bdxi * cdyi - cdxi * bdyi;
      T1 m_b = cdxi * adyi - adxi * cdyi;
      T1 m_c = adxi * bdyi - bdxi * adyi;
      T2 det = T2(alift) * m_a + T2(blift) * m_b + T2(clift) * m_c;
      return (det > 0) ? 1 : (det < 0) ? -1 : 0;
    }
  }

  const T1 lim = T1(1) << 52;
  auto small = [&](T1 v) { return v > -lim && v < lim; };
  if (!std::is_same_v<Int, int32> && small(adxi) && small(adyi) &&
      small(bdxi) && small(bdyi) && small(cdxi) && small(cdyi)) {
    double adx = double(adxi), ady = double(adyi);
    double bdx = double(bdxi), bdy = double(bdyi);
    double cdx = double(cdxi), cdy = double(cdyi);
    double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
    double cdxady = cdx * ady, adxcdy = adx * cdy;
    double adxbdy = adx * bdy, bdxady = bdx * ady;
    double alift = adx * adx + ady * ady;
    double blift = bdx * bdx + bdy * bdy;
    double clift = cdx * cdx + cdy * cdy;
    double det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) +
                 clift * (adxbdy - bdxady);
    auto abs = [](double v) { return v < 0 ? -v : v; };
    double permanent = (abs(bdxcdy) + abs(cdxbdy)) * alift +
                       (abs(cdxady) + abs(adxcdy)) * blift +
                       (abs(adxbdy) + abs(bdxady)) * clift;
    constexpr double eps = 1.1102230246251565e-16; // 2^-53
    constexpr double icc_bound = (10.0 + 96.0 * eps) * eps;
    double errbound = icc_bound * permanent;
    if (det > errbound || det < -errbound)
      return det > 0 ? 1 : -1;
  }

  {
    T2 alift = T2(adxi) * adxi + T2(adyi) * adyi;
    T2 blift = T2(bdxi) * bdxi + T2(bdyi) * bdyi;
    T2 clift = T2(cdxi) * cdxi + T2(cdyi) * cdyi;
    T2 det = alift * (T2(bdxi) * cdyi - T2(bdyi) * cdxi) +
             blift * (T2(cdxi) * adyi - T2(cdyi) * adxi) +
             clift * (T2(adxi) * bdyi - T2(adyi) * bdxi);
    return (det > 0) ? 1 : (det < 0) ? -1 : 0;
  }
}
} // namespace tf::exact
