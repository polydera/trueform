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
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/projection_axes.hpp"
#include <array>
#include <cstddef>

namespace tf::fill {

/// Whether a point known to stand on the line through `a` and `b` stands on
/// the segment between them.
template <typename Int>
auto hole_point_spans(const tf::point<Int, 3> &a, const tf::point<Int, 3> &b,
                      const tf::point<Int, 3> &p) -> bool {
  for (int axis = 0; axis < 3; ++axis) {
    const Int lo = a[axis] < b[axis] ? a[axis] : b[axis];
    const Int hi = a[axis] < b[axis] ? b[axis] : a[axis];
    if (p[axis] < lo || p[axis] > hi)
      return false;
  }
  return true;
}

/// Whether the closed segments `p0 p1` and `q0 q1` share a point. Every
/// contact counts, a shared endpoint included, so a caller that legitimately
/// shares one states that through @ref tf::fill::hole_segments_overrun
/// instead.
template <typename Int>
auto hole_segments_meet(const tf::point<Int, 3> &p0,
                        const tf::point<Int, 3> &p1,
                        const tf::point<Int, 3> &q0,
                        const tf::point<Int, 3> &q1) -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  if (tf::exact::orient3d_sign(p0, p1, q0, q1) != 0)
    return false;

  const T1 ux = T1(p1[0]) - p0[0], uy = T1(p1[1]) - p0[1],
            uz = T1(p1[2]) - p0[2];
  const T1 vx = T1(q1[0]) - q0[0], vy = T1(q1[1]) - q0[1],
            vz = T1(q1[2]) - q0[2];
  const std::array<T2, 3> normal{T2(uy) * vz - T2(uz) * vy,
                                 T2(uz) * vx - T2(ux) * vz,
                                 T2(ux) * vy - T2(uy) * vx};

  if (normal[0] == T2(0) && normal[1] == T2(0) && normal[2] == T2(0)) {
    const T1 wx = T1(q0[0]) - p0[0], wy = T1(q0[1]) - p0[1],
              wz = T1(q0[2]) - p0[2];
    if (T2(uy) * wz - T2(uz) * wy != T2(0) ||
        T2(uz) * wx - T2(ux) * wz != T2(0) ||
        T2(ux) * wy - T2(uy) * wx != T2(0))
      return false;
    return tf::fill::hole_point_spans(p0, p1, q0) ||
           tf::fill::hole_point_spans(p0, p1, q1) ||
           tf::fill::hole_point_spans(q0, q1, p0);
  }

  const auto axes = tf::exact::projection_axes<Int>(normal);
  const auto flat = [&axes](const tf::point<Int, 3> &p) {
    return tf::point<Int, 2>{p[std::size_t(axes.first)],
                             p[std::size_t(axes.second)]};
  };
  const auto f0 = flat(p0), f1 = flat(p1), g0 = flat(q0), g1 = flat(q1);
  const int d0 = tf::exact::orient2d_sign(f0, f1, g0);
  const int d1 = tf::exact::orient2d_sign(f0, f1, g1);
  const int d2 = tf::exact::orient2d_sign(g0, g1, f0);
  const int d3 = tf::exact::orient2d_sign(g0, g1, f1);

  if (d0 == 0 && tf::fill::hole_point_spans(p0, p1, q0))
    return true;
  if (d1 == 0 && tf::fill::hole_point_spans(p0, p1, q1))
    return true;
  if (d2 == 0 && tf::fill::hole_point_spans(q0, q1, p0))
    return true;
  if (d3 == 0 && tf::fill::hole_point_spans(q0, q1, p1))
    return true;
  return d0 * d1 < 0 && d2 * d3 < 0;
}

/// Whether two segments leaving `corner` — one to `p`, one to `q` — share
/// more than that corner, which they do exactly when they run the same way
/// along one line.
template <typename Int>
auto hole_segments_overrun(const tf::point<Int, 3> &corner,
                           const tf::point<Int, 3> &p,
                           const tf::point<Int, 3> &q) -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  const T1 ux = T1(p[0]) - corner[0], uy = T1(p[1]) - corner[1],
            uz = T1(p[2]) - corner[2];
  const T1 vx = T1(q[0]) - corner[0], vy = T1(q[1]) - corner[1],
            vz = T1(q[2]) - corner[2];
  if (T2(uy) * vz - T2(uz) * vy != T2(0) ||
      T2(uz) * vx - T2(ux) * vz != T2(0) ||
      T2(ux) * vy - T2(uy) * vx != T2(0))
    return false;
  return T2(ux) * vx + T2(uy) * vy + T2(uz) * vz > T2(0);
}

} // namespace tf::fill
