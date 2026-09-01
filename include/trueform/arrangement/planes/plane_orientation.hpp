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

#include "../../exact/meta.hpp"
#include <cstddef>

namespace tf::arrangement {

/// The winding a triangulation emits in its own projection — measured, not
/// assumed: the sign of the first non-degenerate triangle's doubled area.
/// One answer per triangulation, so `faces` and `points` are the already
/// materialized tables.
template <typename Int, typename Faces, typename Points>
auto plane_cdt_orientation(const Faces &faces, const Points &points) -> int {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  for (std::size_t t = 0; t < std::size_t(faces.size()); ++t) {
    const auto triangle = faces[t];
    const auto &p0 = points[std::size_t(triangle[0])];
    const auto &p1 = points[std::size_t(triangle[1])];
    const auto &p2 = points[std::size_t(triangle[2])];
    const auto area = T2(T1(p1[0]) - p0[0]) * T2(T1(p2[1]) - p0[1]) -
                      T2(T1(p2[0]) - p0[0]) * T2(T1(p1[1]) - p0[1]);
    if (area != T2(0))
      return area > T2(0) ? 1 : -1;
  }
  return 1;
}

} // namespace tf::arrangement
