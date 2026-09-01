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

#include "../meta.hpp"
#include "./ext_gcd.hpp"
#include "./round_to_wide.hpp"
#include "./wide_axpy.hpp"
#include "./wide_dot.hpp"

#include <array>
#include <cmath>
#include <utility>

namespace tf::exact::door {

/// The plane `N . x = N . p + r` seen from `p`, as a lattice: one
/// Bezout vector with `N . s == 1` and a Lagrange-reduced basis of the
/// kernel lattice. A pure function of the primitive direction `N`.
template <typename Int> struct plane_frame {
  using wide_type = typename tf::exact::meta<Int>::T1;
  std::array<wide_type, 3> s{}, b1{}, b2{};
};

/// The frame of a primitive direction. The reduction is Lagrange's on
/// the two-dimensional kernel lattice: it terminates on its own, and
/// the iteration cap only bounds a basis the double-valued quotient
/// cannot shorten further. A step the rung cannot state ends the
/// reduction on the basis it has, which is a basis either way.
template <typename Int>
auto make_plane_frame(const std::array<typename tf::exact::meta<Int>::T1, 3> &n)
    -> plane_frame<Int> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  plane_frame<Int> frame;
  T1 u(0), v(0);
  const T1 g = ext_gcd(n[0], n[1], u, v);
  if (g == T1(0)) {
    frame.s = {T1(0), T1(0), n[2] > T1(0) ? T1(1) : T1(-1)};
    frame.b1 = {T1(1), T1(0), T1(0)};
    frame.b2 = {T1(0), T1(1), T1(0)};
  } else {
    T1 a(0), b(0);
    ext_gcd(g, n[2], a, b);
    frame.s = {a * u, a * v, b};
    frame.b1 = {-n[1] / g, n[0] / g, T1(0)};
    frame.b2 = {-u * n[2], -v * n[2], g};
  }
  for (int step = 0; step < 24; ++step) {
    if (wide_dot<Int>(frame.b2, frame.b2) < wide_dot<Int>(frame.b1, frame.b1))
      std::swap(frame.b1, frame.b2);
    const T2 length = wide_dot<Int>(frame.b1, frame.b1);
    if (length == T2(0))
      break;
    const double quotient =
        std::round(static_cast<double>(wide_dot<Int>(frame.b1, frame.b2)) /
                   static_cast<double>(length));
    std::array<T1, 3> shortened{};
    if (!wide_axpy<Int>(frame.b2, -round_to_wide<Int>(quotient), frame.b1,
                        shortened))
      break;
    if (wide_dot<Int>(shortened, shortened) >=
        wide_dot<Int>(frame.b2, frame.b2))
      break;
    frame.b2 = shortened;
  }
  return frame;
}

} // namespace tf::exact::door
