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
#include "../../exact/det2_sign.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/projection_axes.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf::fill {

/// Whether the ray from `v` toward `w` lies in the closed wedge the triangle
/// `v a b` spans at `v`, both read in the projection `ax0, ax1` of that
/// triangle's own plane.
template <typename Int>
auto hole_wedge_holds(const tf::point<Int, 3> &v, const tf::point<Int, 3> &a,
                      const tf::point<Int, 3> &b, const tf::point<Int, 3> &w,
                      int ax0, int ax1) -> bool {
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto flat = [ax0, ax1](const tf::point<Int, 3> &p) {
    return tf::point<Int, 2>{p[std::size_t(ax0)], p[std::size_t(ax1)]};
  };
  const auto fv = flat(v), fa = flat(a), fb = flat(b), fw = flat(w);
  const auto along = [&fv](const tf::point<Int, 2> &p,
                           const tf::point<Int, 2> &q) {
    using T1 = typename tf::exact::meta<Int>::T1;
    return T2(T1(p[0]) - fv[0]) * T2(T1(q[0]) - fv[0]) +
           T2(T1(p[1]) - fv[1]) * T2(T1(q[1]) - fv[1]);
  };
  const int span = tf::exact::orient2d_sign(fv, fa, fb);
  const int to_a = tf::exact::orient2d_sign(fv, fa, fw);
  const int to_b = tf::exact::orient2d_sign(fv, fb, fw);
  if (to_a == 0)
    return along(fw, fa) > T2(0);
  if (to_b == 0)
    return along(fw, fb) > T2(0);
  return to_a == span && to_b == -span;
}

/// Whether two triangles of one polygon's triangulation meet anywhere the
/// simplex they legitimately share does not account for.
///
/// The domain is a triangulation of at most five corners, where two
/// triangles always share a corner; a pair that shares none is not admitted.
/// A shared edge decides by coplanarity and the apexes' side of it; a shared
/// corner by the wedge the other triangle's trace on this one's plane points
/// into, which is exact for every crossing because the trace's direction is
/// an affine reading of the two orientations that bracket it.
template <typename Points, typename Index>
auto hole_triangles_overlap(const Points &points,
                            const std::array<Index, 3> &t0,
                            const std::array<Index, 3> &t1) -> bool {
  using Int = std::decay_t<decltype(points[0][0])>;
  using T2 = typename tf::exact::meta<Int>::T2;

  std::array<int, 3> mate{-1, -1, -1};
  int shared = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (t0[std::size_t(i)] == t1[std::size_t(j)]) {
        mate[std::size_t(i)] = j;
        ++shared;
      }
  if (shared != 1 && shared != 2)
    return true;

  const auto at = [&points](Index id) { return points[std::size_t(id)]; };

  if (shared == 2) {
    int apex0 = 0;
    while (mate[std::size_t(apex0)] >= 0)
      ++apex0;
    const auto &u = at(t0[std::size_t((apex0 + 1) % 3)]);
    const auto &v = at(t0[std::size_t((apex0 + 2) % 3)]);
    const auto &w0 = at(t0[std::size_t(apex0)]);
    int apex1 = 0;
    while (t1[std::size_t(apex1)] == t0[std::size_t((apex0 + 1) % 3)] ||
           t1[std::size_t(apex1)] == t0[std::size_t((apex0 + 2) % 3)])
      ++apex1;
    const auto &w1 = at(t1[std::size_t(apex1)]);
    if (tf::exact::orient3d_sign(u, v, w0, w1) != 0)
      return false;
    const auto axes = tf::exact::projection_axes<Int>(u, v, w0);
    const auto flat = [&axes](const tf::point<Int, 3> &p) {
      return tf::point<Int, 2>{p[std::size_t(axes.first)],
                               p[std::size_t(axes.second)]};
    };
    const int s0 = tf::exact::orient2d_sign(flat(u), flat(v), flat(w0));
    const int s1 = tf::exact::orient2d_sign(flat(u), flat(v), flat(w1));
    return s0 == s1;
  }

  int corner0 = 0;
  while (mate[std::size_t(corner0)] < 0)
    ++corner0;
  const auto &v = at(t0[std::size_t(corner0)]);
  const auto &a = at(t0[std::size_t((corner0 + 1) % 3)]);
  const auto &b = at(t0[std::size_t((corner0 + 2) % 3)]);
  const int corner1 = mate[std::size_t(corner0)];
  const auto &c = at(t1[std::size_t((corner1 + 1) % 3)]);
  const auto &d = at(t1[std::size_t((corner1 + 2) % 3)]);

  const auto axes = tf::exact::projection_axes<Int>(v, a, b);
  const auto alpha = tf::exact::orient3d_value(v, a, b, c);
  const auto beta = tf::exact::orient3d_value(v, a, b, d);
  if ((alpha > T2(0) && beta > T2(0)) || (alpha < T2(0) && beta < T2(0)))
    return false;
  if (alpha == T2(0) && beta == T2(0))
    return tf::fill::hole_wedge_holds(v, a, b, c, axes.first, axes.second) ||
           tf::fill::hole_wedge_holds(v, a, b, d, axes.first, axes.second) ||
           tf::fill::hole_wedge_holds(v, c, d, a, axes.first, axes.second) ||
           tf::fill::hole_wedge_holds(v, c, d, b, axes.first, axes.second);
  if (alpha == T2(0))
    return tf::fill::hole_wedge_holds(v, a, b, c, axes.first, axes.second);
  if (beta == T2(0))
    return tf::fill::hole_wedge_holds(v, a, b, d, axes.first, axes.second);

  const auto flat = [&axes](const tf::point<Int, 3> &p) {
    return tf::point<Int, 2>{p[std::size_t(axes.first)],
                             p[std::size_t(axes.second)]};
  };
  const auto fv = flat(v), fa = flat(a), fb = flat(b);
  const auto fc = flat(c), fd = flat(d);
  const bool rising = alpha > T2(0);
  const auto trace = [&alpha, &beta, rising](const T2 &at_c, const T2 &at_d) {
    const int sign = tf::exact::det2_sign<Int>(alpha, at_d, beta, at_c);
    return rising ? sign : -sign;
  };
  const auto along = [&fv](const tf::point<Int, 2> &p,
                           const tf::point<Int, 2> &q) {
    using T1 = typename tf::exact::meta<Int>::T1;
    return T2(T1(p[0]) - fv[0]) * T2(T1(q[0]) - fv[0]) +
           T2(T1(p[1]) - fv[1]) * T2(T1(q[1]) - fv[1]);
  };
  const int span = tf::exact::orient2d_sign(fv, fa, fb);
  const int to_a = trace(tf::exact::orient2d(fv, fa, fc),
                         tf::exact::orient2d(fv, fa, fd));
  const int to_b = trace(tf::exact::orient2d(fv, fb, fc),
                         tf::exact::orient2d(fv, fb, fd));
  if (to_a == 0)
    return trace(along(fc, fa), along(fd, fa)) > 0;
  if (to_b == 0)
    return trace(along(fc, fb), along(fd, fb)) > 0;
  return to_a == span && to_b == -span;
}

} // namespace tf::fill
