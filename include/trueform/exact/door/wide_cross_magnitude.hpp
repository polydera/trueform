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

#include "../meta.hpp"

#include <array>

namespace tf::exact::door {

/// The cross product of two lattice triples measured in its own sum of
/// magnitudes — zero exactly when the two are parallel, and monotone in
/// the independence the rank-2 solve wants. The measure is the sum and
/// not the square because a squared cross leaves the rung the operands
/// were widened to.
template <typename Int>
auto wide_cross_magnitude(
    const std::array<typename tf::exact::meta<Int>::T1, 3> &a,
    const std::array<typename tf::exact::meta<Int>::T1, 3> &b) ->
    typename tf::exact::meta<Int>::T2 {
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto magnitude = [](T2 v) { return v < T2(0) ? -v : v; };
  return magnitude(T2(a[1]) * T2(b[2]) - T2(a[2]) * T2(b[1])) +
         magnitude(T2(a[2]) * T2(b[0]) - T2(a[0]) * T2(b[2])) +
         magnitude(T2(a[0]) * T2(b[1]) - T2(a[1]) * T2(b[0]));
}

} // namespace tf::exact::door
