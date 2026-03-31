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

#include <utility>

namespace tf::exact {

/// Compute 2D projection axes from 3 coplanar points.
///
/// Cross product of (p1-p0) x (p2-p0) gives the face normal.
/// Drop the dominant axis to project into the plane with least distortion.
template <typename Int>
auto projection_axes(const pt3<Int> &p0, const pt3<Int> &p1, const pt3<Int> &p2)
    -> std::pair<int, int> {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 e0x = T1(p1[0]) - p0[0], e0y = T1(p1[1]) - p0[1], e0z = T1(p1[2]) - p0[2];
  T1 e1x = T1(p2[0]) - p0[0], e1y = T1(p2[1]) - p0[1], e1z = T1(p2[2]) - p0[2];
  T2 nx = T2(e0y) * e1z - T2(e0z) * e1y;
  T2 ny = T2(e0z) * e1x - T2(e0x) * e1z;
  T2 nz = T2(e0x) * e1y - T2(e0y) * e1x;
  T2 anx = nx < 0 ? -nx : nx;
  T2 any = ny < 0 ? -ny : ny;
  T2 anz = nz < 0 ? -nz : nz;
  // Swap axes when dominant component is negative to preserve winding.
  if (anz >= anx && anz >= any)
    return nz >= 0 ? std::pair{0, 1} : std::pair{1, 0};
  if (any >= anx)
    return ny >= 0 ? std::pair{2, 0} : std::pair{0, 2};
  return nx >= 0 ? std::pair{1, 2} : std::pair{2, 1};
}

} // namespace tf::exact
