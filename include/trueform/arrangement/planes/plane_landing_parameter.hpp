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

#include "../../exact/meta.hpp"
#include "../../exact/wide_dyadic_ratio.hpp"

namespace tf::arrangement {

/// Where a point lands on the segment `a b` of a plane's projection: the
/// dyadic position of its foot, or `-1` where the foot does not fall
/// strictly inside the span or its position rounds to an end. A landing at
/// an end is a vertex fact, not an edge landing, so it is never stated as
/// one.
template <typename Int, typename A, typename B, typename P>
auto plane_landing_parameter(const A &a, const B &b, const P &p) ->
    typename tf::exact::meta<Int>::param_type {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const T1 dx = T1(b[0]) - T1(a[0]), dy = T1(b[1]) - T1(a[1]);
  const T1 qx = T1(p[0]) - T1(a[0]), qy = T1(p[1]) - T1(a[1]);
  const T2 along = T2(qx) * T2(dx) + T2(qy) * T2(dy);
  const T2 length = T2(dx) * T2(dx) + T2(dy) * T2(dy);
  if (!(along > T2(0)) || !(along < length))
    return param_t(-1);
  const auto parameter = tf::exact::wide_dyadic_ratio<Int>(along, length);
  const param_t whole = param_t(1) << tf::exact::meta<Int>::param_bits;
  if (parameter <= param_t(0) || parameter >= whole)
    return param_t(-1);
  return parameter;
}

} // namespace tf::arrangement
