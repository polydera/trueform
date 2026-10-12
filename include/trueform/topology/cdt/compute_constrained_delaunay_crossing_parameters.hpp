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
#include "../../exact/edge_parameter.hpp"
#include "../../exact/meta.hpp"
#include "./constrained_delaunay_crossing_parameter.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// `meeting` is the crossing itself along the first segment, unrounded: the
/// point the dyadic `self` materializes.
template <typename Int> struct constrained_delaunay_crossing_parameters {
  using param_type = typename tf::exact::meta<Int>::param_type;
  param_type self;
  param_type other;
  tf::exact::edge_parameter<Int> meeting;

  auto valid() const -> bool {
    return self != param_type(-1) && other != param_type(-1);
  }
};

template <typename Owner>
auto compute_constrained_delaunay_crossing_parameters(
    const Owner &owner, typename Owner::index_type first0,
    typename Owner::index_type first1, typename Owner::index_type second0,
    typename Owner::index_type second1)
    -> constrained_delaunay_crossing_parameters<typename Owner::int_type> {
  using Int = typename Owner::int_type;
  using Param = typename Owner::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto a = owner._points[std::size_t(first0)];
  const auto b = owner._points[std::size_t(first1)];
  const auto c = owner._points[std::size_t(second0)];
  const auto d = owner._points[std::size_t(second1)];
  const T2 ux = T2(b[0]) - T2(a[0]);
  const T2 uy = T2(b[1]) - T2(a[1]);
  const T2 vx = T2(d[0]) - T2(c[0]);
  const T2 vy = T2(d[1]) - T2(c[1]);
  const T2 denominator = ux * vy - uy * vx;
  if (denominator == T2(0))
    return {Param(-1), Param(-1), {T2(0), T2(1)}};
  const T2 wx = T2(c[0]) - T2(a[0]);
  const T2 wy = T2(c[1]) - T2(a[1]);
  const T2 numerator = wx * vy - wy * vx;
  return {constrained_delaunay_crossing_parameter<Owner>(numerator,
                                                         denominator),
          constrained_delaunay_crossing_parameter<Owner>(wx * uy - wy * ux,
                                                         denominator),
          denominator < T2(0) ? tf::exact::edge_parameter<Int>{-numerator,
                                                               -denominator}
                              : tf::exact::edge_parameter<Int>{numerator,
                                                               denominator}};
}

} // namespace tf::topology::cdt
