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

#include "../core/polygon.hpp"
#include "./meta.hpp"
#include "./projection_axes.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// Exact sign of signed area for a 2D polygon.
/// Uses T2 accumulator to avoid overflow.
/// Returns: +1 (CCW/face), -1 (CW/hole), 0 (degenerate).
template <typename Policy>
auto signed_area_sign(const tf::polygon<2, Policy> &poly) -> int {
  using Int = tf::coordinate_type<Policy>;
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T2 area2 = 0;
  auto n = poly.size();
  decltype(n) prev = n - 1;
  for (decltype(n) i = 0; i < n; prev = i++) {
    auto &&p0 = poly[prev];
    auto &&p1 = poly[i];
    area2 += T2(T1(p1[1]) + T1(p0[1])) * T2(T1(p0[0]) - T1(p1[0]));
  }
  return (area2 > 0) ? 1 : (area2 < 0) ? -1 : 0;
}

/// Exact signed area (times 2) of a 2D polygon given by local indices.
/// Uses T2 accumulator to avoid overflow.
template <typename Range, typename GetPoint>
auto signed_area_2x(const Range &loop, const GetPoint &get_point)
    -> typename meta<tf::coordinate_type<decltype(get_point(
        loop.front()))>>::T2 {
  using Int = tf::coordinate_type<decltype(get_point(loop.front()))>;
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T2 area2 = 0;
  auto n = loop.size();
  if (n < 3)
    return T2(0);
  pt2<Int> p0 = get_point(loop[n - 1]);
  for (decltype(n) i = 0; i < n; ++i) {
    pt2<Int> p1 = get_point(loop[i]);
    area2 += T2(T1(p1[1]) + T1(p0[1])) * T2(T1(p0[0]) - T1(p1[0]));
    p0 = p1;
  }
  return area2;
}

template <std::size_t Dims, typename Policy>
auto signed_area_2x(const tf::polygon<Dims, Policy> &polygon)
    -> typename meta<tf::coordinate_type<Policy>>::T2 {
  using Int = tf::coordinate_type<Policy>;
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  static_assert(Dims == 2 || Dims == 3);
  if constexpr (Dims == 2) {
    T2 area2 = 0;
    auto n = polygon.size();
    if (n < 3)
      return T2(0);
    pt2<Int> p0 = polygon[n - 1];
    for (decltype(n) i = 0; i < n; ++i) {
      pt2<Int> p1 = polygon[i];
      area2 += T2(T1(p1[1]) + T1(p0[1])) * T2(T1(p0[0]) - T1(p1[0]));
      p0 = p1;
    }
    return area2;
  } else {
    auto axes = tf::exact::projection_axes(polygon[0], polygon[1], polygon[2]);
    auto get_point = [&](const auto &pt) {
      return pt2<Int>{pt[axes.first], pt[axes.second]};
    };
    return signed_area_2x(polygon, get_point);
  }
}

} // namespace tf::exact
