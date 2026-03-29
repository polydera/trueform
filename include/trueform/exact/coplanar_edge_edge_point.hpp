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
template <typename Index>
auto coplanar_edge_edge_point(const vertex<Index> &a0, const vertex<Index> &a1,
                              const vertex<Index> &b0, const vertex<Index> &b1,
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

inline auto coplanar_edge_edge_point(const pt2 &a0, const pt2 &a1,
                                     const pt2 &b0, const pt2 &b1) -> pt2 {
  auto d0 = orient2d(b0, b1, a0);
  auto d1 = orient2d(b0, b1, a1);
  auto abs_d0 = d0 < 0 ? -d0 : d0;
  auto abs_d1 = d1 < 0 ? -d1 : d1;
  auto sum = abs_d0 + abs_d1;
  pt2 point;
  for (int i = 0; i < 2; ++i)
    point[i] = static_cast<int32_t>(
        div_round(abs_d1 * int128(a0[i]) + abs_d0 * int128(a1[i]), sum));
  return point;
}

} // namespace tf::exact
