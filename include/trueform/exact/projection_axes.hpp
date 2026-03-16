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

#include "./int128.hpp"
#include "./vertex.hpp"

#include <utility>

namespace tf::exact {

/// Compute 2D projection axes from 3 coplanar int32 points.
///
/// Cross product of (p1-p0) x (p2-p0) gives the face normal.
/// Drop the dominant axis to project into the plane with least distortion.
inline auto projection_axes(const tf::exact::pt3 &p0, const tf::exact::pt3 &p1,
                            const tf::exact::pt3 &p2) -> std::pair<int, int> {
  using i128 = tf::exact::int128;
  i128 e0x = i128(p1[0]) - p0[0], e0y = i128(p1[1]) - p0[1],
        e0z = i128(p1[2]) - p0[2];
  i128 e1x = i128(p2[0]) - p0[0], e1y = i128(p2[1]) - p0[1],
        e1z = i128(p2[2]) - p0[2];
  i128 nx = e0y * e1z - e0z * e1y;
  i128 ny = e0z * e1x - e0x * e1z;
  i128 nz = e0x * e1y - e0y * e1x;
  if (nx < 0)
    nx = -nx;
  if (ny < 0)
    ny = -ny;
  if (nz < 0)
    nz = -nz;
  if (nz >= nx && nz >= ny)
    return {0, 1};
  if (ny >= nx)
    return {0, 2};
  return {1, 2};
}

} // namespace tf::exact
