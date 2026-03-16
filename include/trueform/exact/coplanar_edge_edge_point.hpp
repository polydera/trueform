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

#include "./orient2d.hpp"
#include "./orient3d.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// Compute crossing point of two coplanar edges via orient2d interpolation.
inline auto coplanar_edge_edge_point(const vertex &a0, const vertex &a1,
                                     const vertex &b0, const vertex &b1,
                                     int ax0, int ax1) -> pt3 {
  pt2 pa0 = {a0.pt[ax0], a0.pt[ax1]}, pa1 = {a1.pt[ax0], a1.pt[ax1]};
  pt2 pb0 = {b0.pt[ax0], b0.pt[ax1]}, pb1 = {b1.pt[ax0], b1.pt[ax1]};
  auto d0 = orient2d(pb0, pb1, pa0);
  auto d1 = orient2d(pb0, pb1, pa1);
  auto abs_d0 = d0 < 0 ? -d0 : d0;
  auto abs_d1 = d1 < 0 ? -d1 : d1;
  auto sum = abs_d0 + abs_d1;
  pt3 point;
  for (int i = 0; i < 3; ++i)
    point[i] = static_cast<int32_t>(div_round(
        abs_d1 * int128(a0.pt[i]) + abs_d0 * int128(a1.pt[i]), sum));
  return point;
}

} // namespace tf::exact
