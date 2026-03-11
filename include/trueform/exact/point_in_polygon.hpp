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

#include "./orient2d.hpp"

namespace tf::exact {

/// Point-in-polygon via crossing number (horizontal ray to +inf).
/// get_point(Index) -> pt2. Returns true if q is strictly inside the polygon.
template <typename Range, typename GetPoint>
auto point_in_polygon(const pt2 &q, const Range &loop,
                      const GetPoint &get_point) -> bool {
  int crossings = 0;
  auto n = loop.size();
  pt2 pi = get_point(loop[n - 1]);
  for (decltype(n) i = 0; i < n; ++i) {
    auto pj = get_point(loop[i]);
    if ((pi[1] <= q[1] && pj[1] > q[1]) || (pj[1] <= q[1] && pi[1] > q[1])) {
      auto o = orient2d(pi, pj, q);
      if (pi[1] < pj[1]) {
        if (o > 0)
          crossings++;
      } else {
        if (o < 0)
          crossings++;
      }
    }
    pi = pj;
  }
  return (crossings & 1) != 0;
}

} // namespace tf::exact
