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
#include "./int128.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// Exact sign of signed area for a 2D polygon with int32 coordinates.
/// Uses int128 accumulator to avoid overflow.
/// Returns: +1 (CCW/face), -1 (CW/hole), 0 (degenerate).
template <typename Policy>
inline auto signed_area_sign(const tf::polygon<2, Policy> &poly) -> int {
  int128 area2 = 0;
  auto n = poly.size();
  decltype(n) prev = n - 1;
  for (decltype(n) i = 0; i < n; prev = i++) {
    auto &&p0 = poly[prev];
    auto &&p1 = poly[i];
    area2 += int128(int64_t(p1[1]) + int64_t(p0[1])) *
             int128(int64_t(p0[0]) - int64_t(p1[0]));
  }
  return (area2 > 0) ? 1 : (area2 < 0) ? -1 : 0;
}

/// Exact signed area (times 2) of a 2D polygon given by local indices.
/// Uses int128 accumulator to avoid overflow.
template <typename Range, typename GetPoint>
auto signed_area_2x(const Range &loop, const GetPoint &get_point) -> int128 {
  int128 area2 = 0;
  auto n = loop.size();
  if (n < 3)
    return 0;
  pt2 p0 = get_point(loop[n - 1]);
  for (decltype(n) i = 0; i < n; ++i) {
    auto p1 = get_point(loop[i]);
    area2 += int128(int64_t(p1[1]) + int64_t(p0[1])) *
             int128(int64_t(p0[0]) - int64_t(p1[0]));
    p0 = p1;
  }
  return area2;
}

template <typename Policy>
auto signed_area_2x(const tf::polygon<2, Policy> &polygon) -> int128 {
  int128 area2 = 0;
  auto n = polygon.size();
  if (n < 3)
    return 0;
  pt2 p0 = polygon[n - 1];
  for (decltype(n) i = 0; i < n; ++i) {
    auto p1 = polygon[i];
    area2 += int128(int64_t(p1[1]) + int64_t(p0[1])) *
             int128(int64_t(p0[0]) - int64_t(p1[0]));
    p0 = p1;
  }
  return area2;
}

} // namespace tf::exact
