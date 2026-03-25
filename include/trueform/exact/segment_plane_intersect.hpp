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

/// Compute crossing point of segment (v0,v1) with plane defined by (a,b,c).
/// Uses orient3d volumes as distance weights, snaps to int32 via div_round.
/// Returns nullopt if both endpoints lie on the plane (degenerate).
/// Caller must ensure v0 and v1 are on opposite sides of the plane.
template <typename Index>
auto segment_plane_intersect(const pt3 &a, const pt3 &b, const pt3 &c,
                             const vertex<Index> &v0, const vertex<Index> &v1)
    -> std::optional<pt3> {
  auto vol_d = orient3d_value(a, b, c, v0.pt);
  auto vol_e = orient3d_value(a, b, c, v1.pt);
  auto abs_d = vol_d < 0 ? -vol_d : vol_d;
  auto abs_e = vol_e < 0 ? -vol_e : vol_e;
  auto sum = abs_d + abs_e;
  if (sum == 0)
    return std::nullopt;
  pt3 point;
  for (int i = 0; i < 3; ++i)
    point[i] = static_cast<int32_t>(
        div_round(abs_e * int128(v0.pt[i]) + abs_d * int128(v1.pt[i]), sum));
  return point;
}

} // namespace tf::exact
