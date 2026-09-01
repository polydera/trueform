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
#include "../../exact/meta.hpp"
#include "./constrained_delaunay_crossing_parameter.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto constrained_delaunay_vertex_parameter(const Owner &owner,
                                           typename Owner::index_type first,
                                           typename Owner::index_type second,
                                           typename Owner::index_type vertex) ->
    typename Owner::param_type {
  using Int = typename Owner::int_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto point_a = owner._points[std::size_t(first)];
  const auto point_b = owner._points[std::size_t(second)];
  const auto point = owner._points[std::size_t(vertex)];
  const T2 delta_x = T2(point_b[0]) - T2(point_a[0]);
  const T2 delta_y = T2(point_b[1]) - T2(point_a[1]);
  const T2 offset_x = T2(point[0]) - T2(point_a[0]);
  const T2 offset_y = T2(point[1]) - T2(point_a[1]);
  return constrained_delaunay_crossing_parameter<Owner>(
      offset_x * delta_x + offset_y * delta_y,
      delta_x * delta_x + delta_y * delta_y);
}

} // namespace tf::topology::cdt
