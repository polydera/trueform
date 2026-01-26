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
#include "./coordinate_type.hpp"
#include "./dot.hpp"
#include "./intersect_status.hpp"
#include "./parallelogram_area.hpp"
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

  const auto scale_den =
      std::abs(d2121) * std::abs(d4343) + std::abs(d4321) * std::abs(d4321);
  auto eps = std::numeric_limits<decltype(denom)>::epsilon() * (1 + scale_den);

  if (std::abs(denom) > eps) {
    t0 = numer / denom;
    t1 = (d1343 + d4321 * t0) / d4343;
    non_parallel = true;
  }

  return std::make_tuple(non_parallel, t0, t1);
}

template <typename Line0, typename Line1>
auto line_line_check_full(const Line0 &line0, const Line1 &line1) {
  using tf::dot;
  using R0 = tf::coordinate_type<Line0>;
  using R1 = tf::coordinate_type<Line1>;
  using Real = tf::coordinate_type<R0, R1>;

  const auto &o0 = line0.origin;
  const auto &d0 = line0.direction;
  const auto &o1 = line1.origin;
  const auto &d1 = line1.direction;

  const auto dif = o0 - o1;
  const Real d1343 = dot(dif, d1);
  const Real d4321 = dot(d1, d0);
  const Real d1321 = dot(dif, d0);
  const Real d4343 = dot(d1, d1);
  const Real d2121 = dot(d0, d0);

  const Real numer = d1343 * d4321 - d1321 * d4343;
  const Real denom = d2121 * d4343 - d4321 * d4321;

  const Real eps = std::numeric_limits<Real>::epsilon();
  const Real scale_den =
      std::abs(d2121) * std::abs(d4343) + std::abs(d4321) * std::abs(d4321);
  const Real eps_den = eps * (scale_den + Real(1));

  const Real NaN = std::numeric_limits<Real>::quiet_NaN();
  Real t0_out = NaN, t1_out = NaN;

  if (std::abs(denom) <= eps_den) {
    const auto area2 = tf::parallelogram_area2(d0, o1 - o0);
    const Real n2 = d2121;
    const Real m2 = dot(o1 - o0, o1 - o0);
    const Real tol2 = eps * (n2 * m2 + Real(1));
    if (area2 <= tol2)
      return std::make_tuple(tf::intersect_status::colinear, t0_out, t1_out);
    else
      return std::make_tuple(tf::intersect_status::parallel, t0_out, t1_out);
  }

  const Real t0 = numer / denom;
  const Real t1 = (d1343 + d4321 * t0) / d4343;

  t0_out = t0;
  t1_out = t1;
  return std::make_tuple(tf::intersect_status::non_parallel, t0_out, t1_out);
}

} // namespace tf::core
