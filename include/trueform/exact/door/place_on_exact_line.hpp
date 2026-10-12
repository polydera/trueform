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

#include "../../core/point.hpp"
#include "../canonical_plane.hpp"
#include "./ext_gcd.hpp"
#include "./pool/exact_axpy.hpp"
#include "./pool/exact_dot.hpp"
#include "./pool/exact_lane.hpp"
#include "./pool/exact_placement_bound.hpp"
#include "./pool/exact_plane_frame.hpp"
#include "./pool/place_on_exact_plane.hpp"
#include "./pool/round_to_exact.hpp"
#include "./pool/wide_to_double.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace tf::exact::door {

/// The frame's Bezout vector moved by kernel vectors until it stands near
/// the plane's own normal: `N . s` stays one, so it is the same frame, and
/// the reach onto the plane it lifts is then as short as the residual. The
/// frame states `s` as its recurrence left it, which for a face the size of
/// the lattice is the product of two cofactors — far longer than a
/// residual times it can carry.
template <typename Int>
auto shorten_bezout(pool::exact_plane_frame<Int> &frame) -> void {
  const double g00 =
      pool::wide_to_double(pool::exact_dot<Int>(frame.b1, frame.b1));
  const double g11 =
      pool::wide_to_double(pool::exact_dot<Int>(frame.b2, frame.b2));
  const double g01 =
      pool::wide_to_double(pool::exact_dot<Int>(frame.b1, frame.b2));
  const double det = g00 * g11 - g01 * g01;
  if (!(det > 0.0))
    return;
  for (int pass = 0; pass < 8; ++pass) {
    const double c0 =
        pool::wide_to_double(pool::exact_dot<Int>(frame.s, frame.b1));
    const double c1 =
        pool::wide_to_double(pool::exact_dot<Int>(frame.s, frame.b2));
    const auto m1 = pool::round_to_exact<Int>((c0 * g11 - c1 * g01) / det);
    const auto m2 = pool::round_to_exact<Int>((c1 * g00 - c0 * g01) / det);
    if (m1 == decltype(m1)(0) && m2 == decltype(m2)(0))
      return;
    auto shortened = frame.s;
    if (!pool::exact_axpy<Int>(shortened, -m1, frame.b1, shortened) ||
        !pool::exact_axpy<Int>(shortened, -m2, frame.b2, shortened))
      return;
    frame.s = shortened;
  }
}

/// The lattice point of two exact planes nearest `p`, ON both.
///
/// The first plane's exact frame puts `p` on it; inside that plane's
/// kernel lattice the second plane is a one-dimensional Diophantine
/// equation, solvable exactly when the gcd of the second normal against
/// the kernel basis divides the residual. The common points then form a
/// coset of one direction, and the answer is its member nearest `p`.
///
/// False when the planes are parallel, when no lattice point lies on both,
/// or when a step of the solve leaves the coefficient rung — a refusal of
/// the bounded search, never a claim that no such point exists.
template <typename Int>
auto place_on_exact_line(const tf::exact::canonical_plane<Int> &a,
                         const tf::exact::canonical_plane<Int> &b,
                         const tf::point<Int, 3> &p, tf::point<Int, 3> &out)
    -> bool {
  using coefficient_type = typename pool::exact_lane<Int>::coefficient_type;
  using product_type = typename pool::exact_lane<Int>::product_type;

  const std::array<coefficient_type, 3> na{a[0], a[1], a[2]};
  const std::array<coefficient_type, 3> nb{b[0], b[1], b[2]};
  auto frame = pool::make_exact_plane_frame<Int>(na);
  if (!frame.available)
    return false;
  shorten_bezout(frame);
  const std::array<coefficient_type, 3> from{
      coefficient_type(p[0]), coefficient_type(p[1]), coefficient_type(p[2])};
  std::array<coefficient_type, 3> step{};
  if (!pool::place_on_exact_plane<Int>(
          frame, -tf::exact::orient3d_plane_value<Int>(a, p), step))
    return false;
  std::array<coefficient_type, 3> on_a{from[0] + step[0], from[1] + step[1],
                                       from[2] + step[2]};

  const product_type along1 = pool::exact_dot<Int>(nb, frame.b1);
  const product_type along2 = pool::exact_dot<Int>(nb, frame.b2);
  if (along1 == product_type(0) && along2 == product_type(0))
    return false;
  const product_type residual =
      product_type(b[3]) - pool::exact_dot<Int>(nb, on_a);
  product_type u(0), v(0);
  const product_type divisor = ext_gcd(along1, along2, u, v);
  if (residual % divisor != product_type(0))
    return false;

  const product_type bound(pool::exact_placement_bound<Int>());
  const auto narrowed = [&bound](const product_type &value,
                                 coefficient_type &to) {
    if (value > bound || value < -bound)
      return false;
    to = static_cast<coefficient_type>(value);
    return true;
  };
  const product_type multiple = residual / divisor;
  coefficient_type toward1(0), toward2(0);
  if (!narrowed(multiple * u, toward1) || !narrowed(multiple * v, toward2) ||
      !pool::exact_axpy<Int>(on_a, toward1, frame.b1, on_a) ||
      !pool::exact_axpy<Int>(on_a, toward2, frame.b2, on_a))
    return false;

  coefficient_type span1(0), span2(0);
  std::array<coefficient_type, 3> along{};
  if (!narrowed(along2 / divisor, span1) || !narrowed(along1 / divisor, span2))
    return false;
  for (std::size_t k = 0; k < 3; ++k)
    if (!narrowed(product_type(span1) * product_type(frame.b1[k]) -
                      product_type(span2) * product_type(frame.b2[k]),
                  along[k]))
      return false;
  const double length =
      pool::wide_to_double(pool::exact_dot<Int>(along, along));
  if (!(length > 0.0))
    return false;
  double toward = 0.0;
  for (std::size_t k = 0; k < 3; ++k)
    toward += pool::wide_to_double(from[k] - on_a[k]) *
              pool::wide_to_double(along[k]);
  const double at = toward / length;
  if (!std::isfinite(at))
    return false;
  const coefficient_type nearest = pool::round_to_exact<Int>(at);

  const coefficient_type ceiling(std::numeric_limits<Int>::max());
  bool found = false;
  product_type best(0);
  std::array<coefficient_type, 3> candidate{}, delta{};
  for (int offset = -2; offset <= 2; ++offset) {
    if (!pool::exact_axpy<Int>(on_a, nearest + coefficient_type(offset), along,
                               candidate))
      continue;
    bool inside = true;
    for (std::size_t k = 0; k < 3; ++k) {
      inside = inside && candidate[k] <= ceiling && candidate[k] >= -ceiling;
      delta[k] = candidate[k] - from[k];
    }
    if (!inside)
      continue;
    const product_type length2 = pool::exact_dot<Int>(delta, delta);
    if (!found || length2 < best) {
      found = true;
      best = length2;
      out = tf::point<Int, 3>{static_cast<Int>(candidate[0]),
                              static_cast<Int>(candidate[1]),
                              static_cast<Int>(candidate[2])};
    }
  }
  return found;
}

} // namespace tf::exact::door
