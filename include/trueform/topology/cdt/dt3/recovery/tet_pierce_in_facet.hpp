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
#include "./tet_projected_point.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Whether the point where the edge d e crosses the plane of `facet` lies
/// in the closed facet, read on the projection `axes`: inside or on one of
/// its triangles. `od` and `oe` are the orient3d volumes of d and e against
/// the plane, strictly of opposite signs; the crossing is read without
/// being formed, each turn about a triangle's edge the affine blend of the
/// turns at d and at e. Every site is read in the decision's `sites`
/// (@ref tet_decision_sites), `od` and `oe` at their depth.
template <typename Sites, typename Facet>
auto tet_pierce_in_facet(
    const Sites &sites, const Facet &facet, const std::pair<int, int> &axes,
    typename Sites::index_type d, typename Sites::index_type e,
    const typename tf::exact::meta<typename Sites::int_type>::T2 &od,
    const typename tf::exact::meta<typename Sites::int_type>::T2 &oe) -> bool {
  using T1 = typename tf::exact::meta<typename Sites::int_type>::T1;
  using T2 = typename tf::exact::meta<typename Sites::int_type>::T2;
  using T3 = typename tf::exact::meta<typename Sites::int_type>::T3;
  const auto xd = tet_projected_point(sites[std::size_t(d)].pt, axes);
  const auto xe = tet_projected_point(sites[std::size_t(e)].pt, axes);
  const auto turn = [](const auto &p, const auto &q, const auto &x) {
    return T2(T1(q[0]) - T1(p[0])) * T2(T1(x[1]) - T1(p[1])) -
           T2(T1(q[1]) - T1(p[1])) * T2(T1(x[0]) - T1(p[0]));
  };
  const int flip = od > oe ? 1 : -1;
  for (const auto &triangle : facet) {
    int positive = 0, negative = 0;
    for (std::size_t k = 0; k < 3; ++k) {
      const auto p =
          tet_projected_point(sites[std::size_t(triangle[k])].pt, axes);
      const auto q = tet_projected_point(
          sites[std::size_t(triangle[(k + 1) % 3])].pt, axes);
      const T3 value = T3(od) * T3(turn(p, q, xe)) - T3(oe) * T3(turn(p, q, xd));
      const int s = (value < T3(0) ? -1 : value > T3(0) ? 1 : 0) * flip;
      positive += s > 0;
      negative += s < 0;
    }
    if (!positive || !negative)
      return true;
  }
  return false;
}

} // namespace tf::topology::cdt::dt3
