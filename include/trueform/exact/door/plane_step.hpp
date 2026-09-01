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
#include "./round_to_wide.hpp"

#include <array>
#include <cmath>

namespace tf::exact::door {

/// The offset step of a named direction: the tolerance measured along
/// the direction's own length, so consecutive offsets of one name stand
/// exactly the tolerance apart in space. Never zero.
template <typename Int>
auto plane_step(const std::array<typename tf::exact::meta<Int>::T1, 3> &normal,
                typename tf::exact::meta<Int>::T1 tolerance) ->
    typename tf::exact::meta<Int>::T1 {
  using T1 = typename tf::exact::meta<Int>::T1;
  const double x = static_cast<double>(normal[0]);
  const double y = static_cast<double>(normal[1]);
  const double z = static_cast<double>(normal[2]);
  const T1 step = round_to_wide<Int>(static_cast<double>(tolerance) *
                                     std::sqrt(x * x + y * y + z * z));
  return step > T1(0) ? step : T1(1);
}

} // namespace tf::exact::door
