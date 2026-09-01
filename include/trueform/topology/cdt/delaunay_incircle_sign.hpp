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
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"

namespace tf::topology::cdt {

/// Exact in-circle sign when the caller has certified that translated lifts
/// and 2x2 minors fit T1. Only the final three products use T2. This is the
/// middle rung between a local native predicate and an all-T2 evaluation.
template <typename Int>
auto delaunay_incircle_sign_t1(const tf::exact::pt2<Int> &a,
                               const tf::exact::pt2<Int> &b,
                               const tf::exact::pt2<Int> &c,
                               const tf::exact::pt2<Int> &d) -> int {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  const T1 adx = T1(a[0]) - T1(d[0]);
  const T1 ady = T1(a[1]) - T1(d[1]);
  const T1 bdx = T1(b[0]) - T1(d[0]);
  const T1 bdy = T1(b[1]) - T1(d[1]);
  const T1 cdx = T1(c[0]) - T1(d[0]);
  const T1 cdy = T1(c[1]) - T1(d[1]);

  const T1 alift = adx * adx + ady * ady;
  const T1 blift = bdx * bdx + bdy * bdy;
  const T1 clift = cdx * cdx + cdy * cdy;
  const T1 minor_a = bdx * cdy - bdy * cdx;
  const T1 minor_b = cdx * ady - cdy * adx;
  const T1 minor_c = adx * bdy - ady * bdx;
  const T2 determinant =
      T2(alift) * minor_a + T2(blift) * minor_b + T2(clift) * minor_c;
  return determinant > T2(0) ? 1 : determinant < T2(0) ? -1 : 0;
}

} // namespace tf::topology::cdt
