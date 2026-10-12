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

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::exact::door {

/// Whether the closed segments `a0 a1` and `b0 b1` share a point, exactly
/// on the lattice: a crossing, an end on the other segment, a shared end or
/// a collinear overlap.
template <typename Int>
auto segments_touch(const tf::point<Int, 3> &a0, const tf::point<Int, 3> &a1,
                    const tf::point<Int, 3> &b0, const tf::point<Int, 3> &b1)
    -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  if (tf::exact::orient3d_sign<Int>(a0, a1, b0, b1) != 0)
    return false;
  const auto cross_zero = [](const tf::point<Int, 3> &o,
                             const tf::point<Int, 3> &p,
                             const tf::point<Int, 3> &q) {
    const std::array<T1, 3> u{T1(p[0]) - T1(o[0]), T1(p[1]) - T1(o[1]),
                              T1(p[2]) - T1(o[2])};
    const std::array<T1, 3> v{T1(q[0]) - T1(o[0]), T1(q[1]) - T1(o[1]),
                              T1(q[2]) - T1(o[2])};
    return T2(u[1]) * T2(v[2]) == T2(u[2]) * T2(v[1]) &&
           T2(u[2]) * T2(v[0]) == T2(u[0]) * T2(v[2]) &&
           T2(u[0]) * T2(v[1]) == T2(u[1]) * T2(v[0]);
  };
  // a segment of no length touches the other where its point lies on it
  const auto on_segment = [&cross_zero](const tf::point<Int, 3> &p,
                                        const tf::point<Int, 3> &s0,
                                        const tf::point<Int, 3> &s1) {
    if (s0 == s1)
      return p == s0;
    if (!cross_zero(s0, s1, p))
      return false;
    for (std::size_t k = 0; k < 3; ++k)
      if (p[k] < std::min(s0[k], s1[k]) || p[k] > std::max(s0[k], s1[k]))
        return false;
    return true;
  };
  if (a0 == a1)
    return on_segment(a0, b0, b1);
  if (b0 == b1)
    return on_segment(b0, a0, a1);
  if (cross_zero(a0, a1, b0) && cross_zero(a0, a1, b1)) {
    // one line: the spans overlap along the axis it moves most on
    std::size_t axis = 0;
    T1 widest(0);
    for (std::size_t k = 0; k < 3; ++k) {
      T1 d = T1(a1[k]) - T1(a0[k]);
      if (d < T1(0))
        d = -d;
      T1 e = T1(b1[k]) - T1(b0[k]);
      if (e < T1(0))
        e = -e;
      if (d + e > widest) {
        widest = d + e;
        axis = k;
      }
    }
    return std::max(std::min(a0[axis], a1[axis]),
                    std::min(b0[axis], b1[axis])) <=
           std::min(std::max(a0[axis], a1[axis]), std::max(b0[axis], b1[axis]));
  }
  {
    // parallel lines that are not one line never meet
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
  const auto between = [](const tf::exact::pt2<Int> &s,
                          const tf::exact::pt2<Int> &e,
                          const tf::exact::pt2<Int> &p) {
    return std::min(s[0], e[0]) <= p[0] && p[0] <= std::max(s[0], e[0]) &&
           std::min(s[1], e[1]) <= p[1] && p[1] <= std::max(s[1], e[1]);
  };
  const T2 o0 = tf::exact::orient2d<Int>(p0, p1, q0);
  const T2 o1 = tf::exact::orient2d<Int>(p0, p1, q1);
  const T2 o2 = tf::exact::orient2d<Int>(q0, q1, p0);
  const T2 o3 = tf::exact::orient2d<Int>(q0, q1, p1);
  if (((o0 > T2(0) && o1 < T2(0)) || (o0 < T2(0) && o1 > T2(0))) &&
      ((o2 > T2(0) && o3 < T2(0)) || (o2 < T2(0) && o3 > T2(0))))
    return true;
  return (o0 == T2(0) && between(p0, p1, q0)) ||
         (o1 == T2(0) && between(p0, p1, q1)) ||
         (o2 == T2(0) && between(q0, q1, p0)) ||
         (o3 == T2(0) && between(q0, q1, p1));
}

} // namespace tf::exact::door
