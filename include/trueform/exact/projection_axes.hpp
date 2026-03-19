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
  int64_t e0x = int64_t(p1[0]) - p0[0], e0y = int64_t(p1[1]) - p0[1],
          e0z = int64_t(p1[2]) - p0[2];
  int64_t e1x = int64_t(p2[0]) - p0[0], e1y = int64_t(p2[1]) - p0[1],
          e1z = int64_t(p2[2]) - p0[2];
  i128 nx = i128(e0y) * e1z - i128(e0z) * e1y;
  i128 ny = i128(e0z) * e1x - i128(e0x) * e1z;
  i128 nz = i128(e0x) * e1y - i128(e0y) * e1x;
  i128 anx = nx < 0 ? -nx : nx;
  i128 any = ny < 0 ? -ny : ny;
  i128 anz = nz < 0 ? -nz : nz;
  // Swap axes when dominant component is negative to preserve winding
  if (anz >= anx && anz >= any)
    return nz > 0 ? std::pair{0, 1} : std::pair{1, 0};
  if (any >= anx)
    return ny > 0 ? std::pair{0, 2} : std::pair{2, 0};
  return nx > 0 ? std::pair{1, 2} : std::pair{2, 1};
}

} // namespace tf::exact
