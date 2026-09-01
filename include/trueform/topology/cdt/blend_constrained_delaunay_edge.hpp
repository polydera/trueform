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
#include "../../core/point.hpp"
#include "../../exact/dyadic_blend.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto blend_constrained_delaunay_edge(const Owner &owner,
                                     typename Owner::index_type first,
                                     typename Owner::index_type second,
                                     typename Owner::param_type parameter)
    -> tf::point<typename Owner::int_type, 2> {
  const auto a = owner._points[std::size_t(first)];
  const auto b = owner._points[std::size_t(second)];
  return {tf::exact::dyadic_blend(a[0], b[0], parameter),
          tf::exact::dyadic_blend(a[1], b[1], parameter)};
}

} // namespace tf::topology::cdt
