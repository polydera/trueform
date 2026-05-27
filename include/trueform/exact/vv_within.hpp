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

/// `|a - b|^2 <= dist_tol_sq`, computed exactly in T2.
template <typename Int>
auto vv_within(const pt3<Int> &a, const pt3<Int> &b,
               typename meta<Int>::T2 dist_tol_sq) -> bool {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 dx = T1(a[0]) - T1(b[0]);
  T1 dy = T1(a[1]) - T1(b[1]);
  T1 dz = T1(a[2]) - T1(b[2]);
  return T2(dx) * dx + T2(dy) * dy + T2(dz) * dz <= dist_tol_sq;
}

template <typename Int>
auto vv_within(const pt2<Int> &a, const pt2<Int> &b,
               typename meta<Int>::T2 dist_tol_sq) -> bool {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 dx = T1(a[0]) - T1(b[0]);
  T1 dy = T1(a[1]) - T1(b[1]);
  return T2(dx) * dx + T2(dy) * dy <= dist_tol_sq;
}

} // namespace tf::exact
