/*
 * Copyright (c) 2026 XLAB
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

#include <cstddef>

namespace tf::exact {

/// Exact `(a - s) . (b - s)` of points carried in one common positive
/// multiple of the `Int` lattice: negative exactly when `s` is strictly
/// inside the ball whose diameter is `ab`, zero on its sphere, positive
/// outside. It scales with the square of the multiple.
template <typename Int, typename Coord>
auto diametral_value(const pt3<Coord> &a, const pt3<Coord> &b,
                     const pt3<Coord> &s) -> typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T2 value(0);
  for (std::size_t i = 0; i < 3; ++i)
    value = value + T2(T1(a[i]) - T1(s[i])) * T2(T1(b[i]) - T1(s[i]));
  return value;
}

} // namespace tf::exact
