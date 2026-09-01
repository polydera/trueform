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

#include "./meta.hpp"
#include "./vertex.hpp"

#include <array>
#include <utility>

namespace tf::exact {

/// The 2D projection axes of a plane given by its exact normal: drop the
/// dominant component to project with least distortion, swapping the pair
/// when that component is negative so the winding is preserved.
template <typename Int>
auto projection_axes(const std::array<typename meta<Int>::T2, 3> &normal)
    -> std::pair<int, int> {
  using T2 = typename meta<Int>::T2;
  const T2 nx = normal[0], ny = normal[1], nz = normal[2];
  const T2 anx = nx < 0 ? -nx : nx;
  const T2 any = ny < 0 ? -ny : ny;
  const T2 anz = nz < 0 ? -nz : nz;
  if (anz >= anx && anz >= any)
    return nz >= 0 ? std::pair{0, 1} : std::pair{1, 0};
  if (any >= anx)
    return ny >= 0 ? std::pair{2, 0} : std::pair{0, 2};
  return nx >= 0 ? std::pair{1, 2} : std::pair{2, 1};
}

/// Compute 2D projection axes from 3 coplanar points.
///
/// Cross product of (p1-p0) x (p2-p0) gives the face normal.
template <typename Int>
auto projection_axes(const pt3<Int> &p0, const pt3<Int> &p1, const pt3<Int> &p2)
    -> std::pair<int, int> {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 e0x = T1(p1[0]) - p0[0], e0y = T1(p1[1]) - p0[1], e0z = T1(p1[2]) - p0[2];
  T1 e1x = T1(p2[0]) - p0[0], e1y = T1(p2[1]) - p0[1], e1z = T1(p2[2]) - p0[2];
  return projection_axes<Int>(std::array<T2, 3>{
      T2(e0y) * e1z - T2(e0z) * e1y, T2(e0z) * e1x - T2(e0x) * e1z,
      T2(e0x) * e1y - T2(e0y) * e1x});
}

/// Projection axes from two crossing edges: normal = (a1-a0) x (b1-b0).
///
/// Robust where the 3-point version degenerates: when an endpoint of one edge
/// lies on the other (a T-junction at a triple point), three edge points are
/// collinear and `projection_axes(a0,a1,b0)` picks the wrong plane. Using both
/// edge directions is non-degenerate for any transversal crossing (zero only
/// if the edges are parallel, which is not a transversal crossing).
template <typename Int>
auto projection_axes_edges(const pt3<Int> &a0, const pt3<Int> &a1,
                           const pt3<Int> &b0, const pt3<Int> &b1)
    -> std::pair<int, int> {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 ax = T1(a1[0]) - a0[0], ay = T1(a1[1]) - a0[1], az = T1(a1[2]) - a0[2];
  T1 bx = T1(b1[0]) - b0[0], by = T1(b1[1]) - b0[1], bz = T1(b1[2]) - b0[2];
  T2 nx = T2(ay) * bz - T2(az) * by;
  T2 ny = T2(az) * bx - T2(ax) * bz;
  T2 nz = T2(ax) * by - T2(ay) * bx;
  T2 anx = nx < 0 ? -nx : nx, any = ny < 0 ? -ny : ny, anz = nz < 0 ? -nz : nz;
  if (anz >= anx && anz >= any)
    return nz >= 0 ? std::pair{0, 1} : std::pair{1, 0};
  if (any >= anx)
    return ny >= 0 ? std::pair{2, 0} : std::pair{0, 2};
  return nx >= 0 ? std::pair{1, 2} : std::pair{2, 1};
}

} // namespace tf::exact
