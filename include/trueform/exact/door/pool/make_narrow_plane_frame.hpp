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

#include "./exact_lane.hpp"
#include "./narrow_lane_bound.hpp"

#include "../../meta.hpp"
#include "../ext_gcd.hpp"
#include "../plane_frame.hpp"
#include "../round_to_wide.hpp"
#include "../wide_axpy.hpp"
#include "../wide_dot.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace tf::exact::door::pool {

/// THE WIDTH GATE. The frame of a primitive EXACT direction stated on the
/// door's own quantized rung, or the refusal that the rung cannot state it.
///
/// It is the door's construction with its initial products guarded: a
/// support normal is a cross of coordinate DIFFERENCES and its Bezout
/// products occupy twice that again, so the components the door forms
/// unguarded — `a u`, `a v`, `-u Nz`, `-v Nz` — are exactly where a normal
/// too wide for this rung wraps. Every one of them is formed on the rung
/// above and admitted only inside
/// @ref tf::exact::door::pool::narrow_lane_bound; a refusal sends the
/// direction to the wide lane.
///
/// Selection is by WIDTH and not by provenance: an exact wall plane whose
/// primitive normal is small stands here beside a quantized name, and a
/// generic triangle's own plane does not.
template <typename Int>
auto make_narrow_plane_frame(
    const std::array<typename exact_lane<Int>::coefficient_type, 3> &n,
    tf::exact::door::plane_frame<Int> &frame) -> bool {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  const T1 bound = narrow_lane_bound<Int>();
  std::array<T1, 3> narrowed{};
  for (std::size_t k = 0; k < 3; ++k) {
    if (n[k] > coefficient_type(bound) || n[k] < -coefficient_type(bound))
      return false;
    narrowed[k] = static_cast<T1>(n[k]);
  }

  const T2 reach(bound);
  const auto narrowed_product = [&reach](const T2 &value, T1 &out) -> bool {
    if (value > reach || value < -reach)
      return false;
    out = static_cast<T1>(value);
    return true;
  };

  T1 u(0), v(0);
  const T1 g = tf::exact::door::ext_gcd(narrowed[0], narrowed[1], u, v);
  if (g == T1(0)) {
    if (narrowed[2] == T1(0))
      return false;
    frame.s = {T1(0), T1(0), narrowed[2] > T1(0) ? T1(1) : T1(-1)};
    frame.b1 = {T1(1), T1(0), T1(0)};
    frame.b2 = {T1(0), T1(1), T1(0)};
  } else {
    T1 a(0), b(0);
    tf::exact::door::ext_gcd(g, narrowed[2], a, b);
    if (!narrowed_product(T2(a) * T2(u), frame.s[0]) ||
        !narrowed_product(T2(a) * T2(v), frame.s[1]) ||
        !narrowed_product(T2(b), frame.s[2]) ||
        !narrowed_product(-T2(narrowed[1] / g), frame.b1[0]) ||
        !narrowed_product(T2(narrowed[0] / g), frame.b1[1]) ||
        !narrowed_product(-(T2(u) * T2(narrowed[2])), frame.b2[0]) ||
        !narrowed_product(-(T2(v) * T2(narrowed[2])), frame.b2[1]) ||
        !narrowed_product(T2(g), frame.b2[2]))
      return false;
    frame.b1[2] = T1(0);
  }

  for (int step = 0; step < 24; ++step) {
    if (tf::exact::door::wide_dot<Int>(frame.b2, frame.b2) <
        tf::exact::door::wide_dot<Int>(frame.b1, frame.b1))
      std::swap(frame.b1, frame.b2);
    const T2 length = tf::exact::door::wide_dot<Int>(frame.b1, frame.b1);
    if (length == T2(0))
      break;
    const double quotient = std::round(
        static_cast<double>(tf::exact::door::wide_dot<Int>(frame.b1, frame.b2)) /
        static_cast<double>(length));
    if (!std::isfinite(quotient))
      break;
    std::array<T1, 3> shortened{};
    if (!tf::exact::door::wide_axpy<Int>(
            frame.b2, -tf::exact::door::round_to_wide<Int>(quotient), frame.b1,
            shortened))
      break;
    if (tf::exact::door::wide_dot<Int>(shortened, shortened) >=
        tf::exact::door::wide_dot<Int>(frame.b2, frame.b2))
      break;
    frame.b2 = shortened;
  }
  return true;
}

} // namespace tf::exact::door::pool
