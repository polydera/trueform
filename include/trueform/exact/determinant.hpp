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

/// Exact 3x3 determinant det(a, b, c) of three 3D points.
/// Returns the scalar triple product a . (b x c).
template <typename Int>
auto determinant_value(const pt3<Int> &a, const pt3<Int> &b,
                       const pt3<Int> &c) -> typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 b1c2 = T1(b[1]) * c[2], b2c1 = T1(b[2]) * c[1];
  T1 b2c0 = T1(b[2]) * c[0], b0c2 = T1(b[0]) * c[2];
  T1 b0c1 = T1(b[0]) * c[1], b1c0 = T1(b[1]) * c[0];
  return T2(a[0]) * (T2(b1c2) - b2c1) + T2(a[1]) * (T2(b2c0) - b0c2) +
         T2(a[2]) * (T2(b0c1) - b1c0);
}

/// Exact sign of det(a, b, c). Returns -1, 0, or +1.
template <typename Int>
auto determinant_sign(const pt3<Int> &a, const pt3<Int> &b,
                      const pt3<Int> &c) -> int {
  auto val = determinant_value(a, b, c);
  return (val > 0) ? 1 : (val < 0) ? -1 : 0;
}

} // namespace tf::exact
