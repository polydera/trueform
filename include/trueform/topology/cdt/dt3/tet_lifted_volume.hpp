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
#include "../../../exact/meta.hpp"
#include "../../../exact/orient3d.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Twenty-four times the volume of the prism between the finite cell
/// `c0 c1 c2 c3` and its lift onto the paraboloid, its corners read in the
/// decision's `sites` (@ref tet_decision_sites): six times its volume, taken
/// positive, times the sum of its corners' squared norms, exact on the
/// sites' `T3`. A Delaunay complex minimises the sum over its cells, so a
/// flip lowers the sum exactly when it moves the complex toward Delaunay —
/// the objective `recoverdelaunay` keeps.
template <typename Sites>
auto tet_lifted_volume(const Sites &sites, typename Sites::index_type c0,
                       typename Sites::index_type c1,
                       typename Sites::index_type c2,
                       typename Sites::index_type c3) ->
    typename tf::exact::meta<typename Sites::int_type>::T3 {
  using Int = typename Sites::int_type;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using T3 = typename tf::exact::meta<Int>::T3;
  const auto p0 = sites[std::size_t(c0)].pt;
  const auto p1 = sites[std::size_t(c1)].pt;
  const auto p2 = sites[std::size_t(c2)].pt;
  const auto p3 = sites[std::size_t(c3)].pt;
  T2 six = tf::exact::orient3d_value_scaled<Int>(p0, p1, p2, p3);
  if (six < 0)
    six = -six;
  const auto weight = [](const auto &p) {
    T2 w(0);
    for (std::size_t k = 0; k < 3; ++k) {
      const T1 c = T1(p[k]);
      w += T2(c) * T2(c);
    }
    return w;
  };
  const T2 w = weight(p0) + weight(p1) + weight(p2) + weight(p3);
  return T3(six) * T3(w);
}

} // namespace tf::topology::cdt::dt3
