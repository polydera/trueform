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
 * Author: Ziga Sajovic
 */
#pragma once

#include <algorithm>
#include "../core/containment.hpp"
#include "../core/polygon.hpp"
#include "./orient2d.hpp"

namespace tf::exact {

/// Exact point-in-polygon classification via winding number.
/// Uses orient2d with int128 — no epsilon, no floating point.
/// Returns inside, outside, or on_boundary.
template <typename Policy>
auto classify(const tf::point<int32_t, 2> &q,
              const tf::polygon<2, Policy> &poly) -> tf::containment {
  int crossings = 0;
  auto n = poly.size();
  pt2 pi = {poly[n - 1][0], poly[n - 1][1]};
  for (decltype(n) i = 0; i < n; ++i) {
    pt2 pj = {poly[i][0], poly[i][1]};

    // On-boundary check: orient2d == 0 and within bounding box
    auto o = orient2d(pi, pj, {q[0], q[1]});
    if (o == 0) {
      if (q[0] >= std::min(pi[0], pj[0]) && q[0] <= std::max(pi[0], pj[0]) &&
          q[1] >= std::min(pi[1], pj[1]) && q[1] <= std::max(pi[1], pj[1]))
        return tf::containment::on_boundary;
    }

    // Winding number crossing test
    if ((pi[1] <= q[1] && pj[1] > q[1]) ||
        (pj[1] <= q[1] && pi[1] > q[1])) {
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
  return (crossings & 1) != 0 ? tf::containment::inside
                               : tf::containment::outside;
}

} // namespace tf::exact
