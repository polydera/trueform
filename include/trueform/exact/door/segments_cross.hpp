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

#include "../../core/point.hpp"
#include "../meta.hpp"
#include "../orient2d.hpp"
#include "../orient3d.hpp"
#include "../projection_axes.hpp"
#include "../vertex.hpp"

#include <array>
#include <cstddef>

namespace tf::exact::door {

/// Whether the segments `a0 a1` and `b0 b1` meet at one point interior to
/// both, exactly on the lattice. A shared end, an end on the other segment
/// and a collinear overlap are not crossings: each is an end standing on the
/// other, which is a vertex on a closed face.
template <typename Int>
auto segments_cross(const tf::point<Int, 3> &a0, const tf::point<Int, 3> &a1,
                    const tf::point<Int, 3> &b0, const tf::point<Int, 3> &b1)
    -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  if (a0 == a1 || b0 == b1 ||
      tf::exact::orient3d_sign<Int>(a0, a1, b0, b1) != 0)
    return false;
  {
    const std::array<T1, 3> u{T1(a1[0]) - T1(a0[0]), T1(a1[1]) - T1(a0[1]),
                              T1(a1[2]) - T1(a0[2])};
    const std::array<T1, 3> v{T1(b1[0]) - T1(b0[0]), T1(b1[1]) - T1(b0[1]),
                              T1(b1[2]) - T1(b0[2])};
    if (T2(u[1]) * T2(v[2]) == T2(u[2]) * T2(v[1]) &&
        T2(u[2]) * T2(v[0]) == T2(u[0]) * T2(v[2]) &&
        T2(u[0]) * T2(v[1]) == T2(u[1]) * T2(v[0]))
      return false;
  }
  const auto [x, y] = tf::exact::projection_axes_edges<Int>(a0, a1, b0, b1);
  const auto flat = [x = x, y = y](const tf::point<Int, 3> &p) {
    return tf::exact::pt2<Int>{p[std::size_t(x)], p[std::size_t(y)]};
  };
  const auto p0 = flat(a0), p1 = flat(a1), q0 = flat(b0), q1 = flat(b1);
  const T2 o0 = tf::exact::orient2d<Int>(p0, p1, q0);
  const T2 o1 = tf::exact::orient2d<Int>(p0, p1, q1);
  const T2 o2 = tf::exact::orient2d<Int>(q0, q1, p0);
  const T2 o3 = tf::exact::orient2d<Int>(q0, q1, p1);
  return ((o0 > T2(0) && o1 < T2(0)) || (o0 < T2(0) && o1 > T2(0))) &&
         ((o2 > T2(0) && o3 < T2(0)) || (o2 < T2(0) && o3 > T2(0)));
}

} // namespace tf::exact::door
