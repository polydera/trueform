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

#include "./orient3d.hpp"
#include <optional>

namespace tf::exact {

/// Compute crossing point of segment (p0,p1) with plane defined by (a,b,c).
/// Uses orient3d volumes as distance weights, snaps to Int via div_round.
/// Returns nullopt if both endpoints lie on the plane (degenerate).
/// Caller must ensure p0 and p1 are on opposite sides of the plane.
template <typename Int>
auto segment_plane_intersect(const pt3<Int> &a, const pt3<Int> &b,
                             const pt3<Int> &c, const pt3<Int> &p0,
                             const pt3<Int> &p1) -> std::optional<pt3<Int>> {
  using T2 = typename meta<Int>::T2;
  auto vol_d = orient3d_value(a, b, c, p0);
  auto vol_e = orient3d_value(a, b, c, p1);
  auto abs_d = vol_d < 0 ? -vol_d : vol_d;
  auto abs_e = vol_e < 0 ? -vol_e : vol_e;
  auto sum = abs_d + abs_e;
  if (sum == 0)
    return std::nullopt;
  pt3<Int> point;
  for (int i = 0; i < 3; ++i)
    point[i] = static_cast<Int>(
        div_round(abs_e * T2(p0[i]) + abs_d * T2(p1[i]), sum));
  return point;
}

template <typename Index, typename Int>
auto segment_plane_intersect(const pt3<Int> &a, const pt3<Int> &b,
                             const pt3<Int> &c, const vertex<Index, Int> &v0,
                             const vertex<Index, Int> &v1)
    -> std::optional<pt3<Int>> {
  return segment_plane_intersect(a, b, c, v0.pt, v1.pt);
}

} // namespace tf::exact
