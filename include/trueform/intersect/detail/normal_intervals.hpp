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

#include "../../core/intersects.hpp"
#include "../../core/interval.hpp"
#include "../../core/policy/normal.hpp"
#include "../../core/polygon.hpp"

namespace tf::intersect {
template <std::size_t Dims, typename Policy0, typename Policy1>
auto normal_intervals(const tf::polygon<Dims, Policy0> &_poly0,
                      const tf::polygon<Dims, Policy1> &_poly1) {
  auto poly0 = tf::tag_normal(_poly0);
  auto poly1 = tf::tag_normal(_poly1);
  {
    auto r0 =
        tf::make_interval(poly0, tf::make_line_like(poly0[0], poly0.normal()));
    auto r1 =
        tf::make_interval(poly1, tf::make_line_like(poly0[0], poly0.normal()));
    if (!tf::intersects(r0, r1))
      return false;
  }
  {
    auto r0 =
        tf::make_interval(poly0, tf::make_line_like(poly1[0], poly1.normal()));
    auto r1 =
        tf::make_interval(poly1, tf::make_line_like(poly1[0], poly1.normal()));
    if (!tf::intersects(r0, r1))
      return false;
  }
  return true;
}
} // namespace tf::intersect
