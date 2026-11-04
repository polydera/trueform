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
#include "../../../../exact/meta.hpp"
#include "./tet_orient2d_sign.hpp"
#include "./tet_projected_point.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Whether the triangle of the sites `key`, lying in the plane of `facet`
/// and crossing none of its rim's edges, lies in the closed facet, read on
/// the projection `axes`: its corners and its centroid each inside or on
/// one of the facet's triangles. A triangle whose corners stand on a
/// non-convex facet may bridge a notch of it, and its centroid then stands
/// outside. Every site is read in the decision's `sites`
/// (@ref tet_decision_sites).
template <typename Sites, typename Facet, typename Key>
auto tet_triangle_in_facet(const Sites &sites, const Facet &facet,
                           const std::pair<int, int> &axes, const Key &key)
    -> bool {
  using Int = typename Sites::int_type;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto flat = [&sites, &axes](std::size_t site) {
    return tet_projected_point(sites[site].pt, axes);
  };
  const auto inside = [&facet, &flat](const auto &sign) {
    for (const auto &triangle : facet) {
      int positive = 0, negative = 0;
      for (std::size_t e = 0; e < 3; ++e) {
        const int s = sign(flat(std::size_t(triangle[e])),
                           flat(std::size_t(triangle[e == 2 ? 0 : e + 1])));
        positive += s > 0;
        negative += s < 0;
      }
      if (!positive || !negative)
        return true;
    }
    return false;
  };
  std::array<T1, 2> centre{T1(0), T1(0)};
  for (auto site : key) {
    const auto p = flat(std::size_t(site));
    if (!inside([&p](const auto &a, const auto &b) {
          return tet_orient2d_sign<Int>(a, b, p);
        }))
      return false;
    centre[0] = centre[0] + T1(p[0]);
    centre[1] = centre[1] + T1(p[1]);
  }
  return inside([&centre](const auto &a, const auto &b) {
    const T1 ax = T1(3) * T1(a[0]), ay = T1(3) * T1(a[1]);
    const T2 value = T2(T1(3) * T1(b[0]) - ax) * T2(centre[1] - ay) -
                     T2(T1(3) * T1(b[1]) - ay) * T2(centre[0] - ax);
    return value > T2(0) ? 1 : value < T2(0) ? -1 : 0;
  });
}

} // namespace tf::topology::cdt::dt3
