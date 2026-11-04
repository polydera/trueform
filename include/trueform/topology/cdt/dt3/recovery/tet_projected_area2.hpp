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
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Twice the area of the triangle of sites `x`, `y`, `z` of the decision's
/// `sites` (@ref tet_decision_sites) read on the projection `axes` names,
/// unsigned and exact on their lattice.
template <typename Sites>
auto tet_projected_area2(const Sites &sites, const std::pair<int, int> &axes,
                         typename Sites::index_type x,
                         typename Sites::index_type y,
                         typename Sites::index_type z) ->
    typename tf::exact::meta<typename Sites::int_type>::T2 {
  using T1 = typename tf::exact::meta<typename Sites::int_type>::T1;
  using T2 = typename tf::exact::meta<typename Sites::int_type>::T2;
  const auto p = tet_projected_point(sites[std::size_t(x)].pt, axes);
  const auto q = tet_projected_point(sites[std::size_t(y)].pt, axes);
  const auto s = tet_projected_point(sites[std::size_t(z)].pt, axes);
  const T2 value = T2(T1(q[0]) - T1(p[0])) * T2(T1(s[1]) - T1(p[1])) -
                   T2(T1(q[1]) - T1(p[1])) * T2(T1(s[0]) - T1(p[0]));
  return value < T2(0) ? -value : value;
}

} // namespace tf::topology::cdt::dt3
