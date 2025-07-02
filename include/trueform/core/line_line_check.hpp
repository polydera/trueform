/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./dot.hpp"
#include "./coordinate_type.hpp"
#include <tuple>

namespace tf::core {

template <typename Line0, typename Line1>
auto line_line_check(const Line0 &line0, const Line1 &line1) {
  auto dif = line0.origin - line1.origin;

  auto d1343 = tf::dot(dif, line1.direction);
  auto d4321 = tf::dot(line1.direction, line0.direction);
  auto d1321 = tf::dot(dif, line0.direction);
  auto d4343 = tf::dot(line1.direction, line1.direction);
  auto d2121 = tf::dot(line0.direction, line0.direction);

  auto numer = d1343 * d4321 - d1321 * d4343;
  auto denom = d2121 * d4343 - d4321 * d4321;

  using real_t = tf::coordinate_type<decltype(numer), decltype(denom)>;
  real_t t0 = real_t(0);
  real_t t1 = real_t(0);
  bool non_parallel = false;

  auto eps = std::numeric_limits<decltype(denom)>::epsilon() *
             (std::abs(d2121) + std::abs(d4343));

  if (std::abs(denom) > eps) {
    t0 = numer / denom;
    t1 = (d1343 + d4321 * t0) / d4343;
    non_parallel = true;
  }

  return std::make_tuple(non_parallel, t0, t1);
}
} // namespace tf::implementation
