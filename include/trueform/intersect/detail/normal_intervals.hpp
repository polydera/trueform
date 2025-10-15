/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
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
