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
template <typename Int>
auto coplanar_edge_edge_point(const pt3<Int> &a0, const pt3<Int> &a1,
                              const pt3<Int> &b0, const pt3<Int> &b1,
                              int ax0, int ax1) -> pt3<Int> {
  using T2 = typename meta<Int>::T2;
  pt2<Int> pa0 = {a0[ax0], a0[ax1]}, pa1 = {a1[ax0], a1[ax1]};
  pt2<Int> pb0 = {b0[ax0], b0[ax1]}, pb1 = {b1[ax0], b1[ax1]};
  auto d0 = orient2d(pb0, pb1, pa0);
  auto d1 = orient2d(pb0, pb1, pa1);
  auto abs_d0 = d0 < 0 ? -d0 : d0;
  auto abs_d1 = d1 < 0 ? -d1 : d1;
  auto sum = abs_d0 + abs_d1;
  pt3<Int> point;
  for (int i = 0; i < 3; ++i)
    point[i] = static_cast<Int>(
        div_round(abs_d1 * T2(a0[i]) + abs_d0 * T2(a1[i]), sum));
  return point;
}

template <typename Index, typename Int>
auto coplanar_edge_edge_point(const vertex<Index, Int> &a0,
                              const vertex<Index, Int> &a1,
                              const vertex<Index, Int> &b0,
                              const vertex<Index, Int> &b1, int ax0, int ax1)
    -> pt3<Int> {
  return coplanar_edge_edge_point(a0.pt, a1.pt, b0.pt, b1.pt, ax0, ax1);
}

template <typename Int>
auto coplanar_edge_edge_point(const pt2<Int> &a0, const pt2<Int> &a1,
                              const pt2<Int> &b0, const pt2<Int> &b1)
    -> pt2<Int> {
  using T2 = typename meta<Int>::T2;
  auto d0 = orient2d(b0, b1, a0);
  auto d1 = orient2d(b0, b1, a1);
  auto abs_d0 = d0 < 0 ? -d0 : d0;
  auto abs_d1 = d1 < 0 ? -d1 : d1;
  auto sum = abs_d0 + abs_d1;
  pt2<Int> point;
  for (int i = 0; i < 2; ++i)
    point[i] = static_cast<Int>(
        div_round(abs_d1 * T2(a0[i]) + abs_d0 * T2(a1[i]), sum));
  return point;
}

} // namespace tf::exact
