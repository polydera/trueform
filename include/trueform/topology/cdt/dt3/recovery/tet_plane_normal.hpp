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
#include "../../../../exact/meta.hpp"
#include "../../../../exact/vertex.hpp"
#include <array>

namespace tf::topology::cdt::dt3 {

/// The exact normal of the plane through three points of the `Int` lattice
/// carried in `Coord`, `(b - a) x (c - a)` on `meta<Int>::T2`; zero when
/// they stand on one line.
template <typename Int, typename Coord>
auto tet_plane_normal(const tf::exact::pt3<Coord> &a,
                      const tf::exact::pt3<Coord> &b,
                      const tf::exact::pt3<Coord> &c)
    -> std::array<typename tf::exact::meta<Int>::T2, 3> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  const T1 u[3] = {T1(b[0]) - a[0], T1(b[1]) - a[1], T1(b[2]) - a[2]};
  const T1 v[3] = {T1(c[0]) - a[0], T1(c[1]) - a[1], T1(c[2]) - a[2]};
  return {T2(u[1]) * v[2] - T2(u[2]) * v[1], T2(u[2]) * v[0] - T2(u[0]) * v[2],
          T2(u[0]) * v[1] - T2(u[1]) * v[0]};
}

} // namespace tf::topology::cdt::dt3
