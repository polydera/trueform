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
#include "../../../../exact/vertex.hpp"
#include "./tet_projected_point.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Whether the point `pt`, on the plane of `facet` and read beside the
/// decision's `sites` (@ref tet_decision_sites), lies in it off its rim,
/// read on the projection `axes`: inside a triangle or on an edge two of
/// them hold, never on an edge one holds nor at a corner.
template <typename Sites, typename Facet>
auto tet_point_in_facet(const Sites &sites, const Facet &facet,
                        const std::pair<int, int> &axes,
                        const tf::exact::pt3<typename Sites::coord_type> &pt)
    -> bool {
  using Coord = typename Sites::coord_type;
  using T1 = typename tf::exact::meta<typename Sites::int_type>::T1;
  using T2 = typename tf::exact::meta<typename Sites::int_type>::T2;
  const auto x = tet_projected_point(pt, axes);
  const auto turn = [&x](const tf::exact::pt2<Coord> &p,
                         const tf::exact::pt2<Coord> &q) {
    const T2 value = T2(T1(q[0]) - T1(p[0])) * T2(T1(x[1]) - T1(p[1])) -
                     T2(T1(q[1]) - T1(p[1])) * T2(T1(x[0]) - T1(p[0]));
    return value < T2(0) ? -1 : value > T2(0) ? 1 : 0;
  };
  for (const auto &triangle : facet) {
    int positive = 0, negative = 0;
    std::size_t on = 0;
    for (std::size_t e = 0; e < 3; ++e) {
      const int s =
          turn(tet_projected_point(sites[std::size_t(triangle[e])].pt, axes),
               tet_projected_point(sites[std::size_t(triangle[(e + 1) % 3])].pt,
                                   axes));
      positive += s > 0;
      negative += s < 0;
      on = s ? on : e;
    }
    if (positive && negative)
      continue;
    if (positive + negative == 3)
      return true;
    if (positive + negative < 2)
      return false;
    const auto p = triangle[on], q = triangle[(on + 1) % 3];
    std::size_t holders = 0;
    for (const auto &other : facet) {
      bool has_p = false, has_q = false;
      for (std::size_t k = 0; k < 3; ++k) {
        has_p = has_p || other[k] == p;
        has_q = has_q || other[k] == q;
      }
      holders += has_p && has_q;
    }
    return holders > 1;
  }
  return false;
}

} // namespace tf::topology::cdt::dt3
