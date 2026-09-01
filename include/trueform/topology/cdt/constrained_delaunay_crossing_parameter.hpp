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
#include "../../exact/dyadic_ratio.hpp"
#include "../../exact/meta.hpp"

namespace tf::topology::cdt {

/// Test interiority on the exact ratio before rounding it to the dyadic
/// parameter grid. Rounding is clamped to the nearest interior grid value, so
/// an accepted crossing never becomes an endpoint.
template <typename Owner>
auto constrained_delaunay_crossing_parameter(
    typename tf::exact::meta<typename Owner::int_type>::T2 numerator,
    typename tf::exact::meta<typename Owner::int_type>::T2 denominator) ->
    typename Owner::param_type {
  using Int = typename Owner::int_type;
  using Param = typename Owner::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  if (denominator < T2(0)) {
    numerator = -numerator;
    denominator = -denominator;
  }
  if (numerator <= T2(0) || denominator <= numerator)
    return Param(-1);
  const Param parameter = tf::exact::dyadic_ratio<Int>(numerator, denominator);
  const Param maximum = Param(1) << Owner::crossing_param_bits;
  return parameter <= Param(0)
             ? Param(1)
             : (parameter >= maximum ? maximum - Param(1) : parameter);
}

} // namespace tf::topology::cdt
