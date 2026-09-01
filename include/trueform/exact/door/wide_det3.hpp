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

/// The determinant of three lattice triples, on the rung above them.
template <typename Int>
auto wide_det3(const std::array<typename tf::exact::meta<Int>::T1, 3> &a,
               const std::array<typename tf::exact::meta<Int>::T1, 3> &b,
               const std::array<typename tf::exact::meta<Int>::T1, 3> &c) ->
    typename tf::exact::meta<Int>::T2 {
  using T2 = typename tf::exact::meta<Int>::T2;
  return T2(a[0]) * (T2(b[1]) * T2(c[2]) - T2(b[2]) * T2(c[1])) -
         T2(a[1]) * (T2(b[0]) * T2(c[2]) - T2(b[2]) * T2(c[0])) +
         T2(a[2]) * (T2(b[0]) * T2(c[1]) - T2(b[1]) * T2(c[0]));
}

} // namespace tf::exact::door
