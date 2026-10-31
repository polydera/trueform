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
#include <cstddef>

namespace tf::exact::door {

/// A point's squared distance to an edge's line, exactly, as the fraction
/// `cross2 / length2` with `cross2 = |(v - a) x (b - a)|^2` and
/// `length2 = |b - a|^2`.
template <typename Int> struct edge_span_distance {
  typename tf::exact::meta<Int>::T3 cross2;
  typename tf::exact::meta<Int>::T2 length2;
};

/// Whether `x` is strictly nearer its edge than `y` is to its own:
/// `x.cross2 * y.length2 < y.cross2 * x.length2`, on the construction rung.
template <typename Int>
auto nearer_edge_span(const edge_span_distance<Int> &x,
                      const edge_span_distance<Int> &y) -> bool {
  using T3 = typename tf::exact::meta<Int>::T3;
  return x.cross2 * T3(y.length2) < y.cross2 * T3(x.length2);
}

/// Whether `v` stands within `band` of the OPEN span of the edge (a, b):
/// its projection falls strictly between the ends,
/// `0 < (v - a).(b - a) < |b - a|^2`, and its distance to the line is at
/// most the band, `|(v - a) x (b - a)|^2 <= band^2 |b - a|^2`. Beyond the
/// span is not on the edge, and an end is a corner, which is not this
/// question. `distance` receives the exact distance when the answer is yes.
template <typename Int>
auto within_band_of_edge_span(const tf::point<Int, 3> &v,
                              const tf::point<Int, 3> &a,
                              const tf::point<Int, 3> &b, Int band,
                              edge_span_distance<Int> &distance) -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using T3 = typename tf::exact::meta<Int>::T3;
  const T1 dx = T1(b[0]) - T1(a[0]), dy = T1(b[1]) - T1(a[1]),
           dz = T1(b[2]) - T1(a[2]);
  const T1 wx = T1(v[0]) - T1(a[0]), wy = T1(v[1]) - T1(a[1]),
           wz = T1(v[2]) - T1(a[2]);
  const T2 dot = T2(dx) * T2(wx) + T2(dy) * T2(wy) + T2(dz) * T2(wz);
  const T2 length2 = T2(dx) * T2(dx) + T2(dy) * T2(dy) + T2(dz) * T2(dz);
  if (!(dot > T2(0) && dot < length2))
    return false;
  const T2 cx = T2(dy) * T2(wz) - T2(dz) * T2(wy);
  const T2 cy = T2(dz) * T2(wx) - T2(dx) * T2(wz);
  const T2 cz = T2(dx) * T2(wy) - T2(dy) * T2(wx);
  const T3 cross2 = T3(cx) * T3(cx) + T3(cy) * T3(cy) + T3(cz) * T3(cz);
  if (cross2 > T3(T2(band) * T2(band)) * T3(length2))
    return false;
  distance = {cross2, length2};
  return true;
}

} // namespace tf::exact::door
