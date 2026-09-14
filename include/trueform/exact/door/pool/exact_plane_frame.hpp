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

#include "./exact_dot.hpp"
#include "./exact_lane.hpp"
#include "./exact_placement_bound.hpp"
#include "./reduce_on_product_rung.hpp"

#include "../ext_gcd.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace tf::exact::door::pool {

/// The plane `N . x = N . p + r` seen from `p`, as a lattice: one Bezout
/// vector with `N . s == 1` and a Lagrange-reduced basis of the kernel
/// lattice. A pure function of the primitive EXACT direction `N`.
///
/// `available` is false only when the REDUCED frame does not stand on the
/// coefficient rung. What the construction starts from does not: the
/// initial kernel vector is a product of two Bezout cofactors and is a rung
/// above the coefficient one BY CONSTRUCTION, so a bound taken before the
/// reduction refuses exactly the skew kernels the reduction exists to fix.
template <typename Int> struct exact_plane_frame {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  std::array<coefficient_type, 3> s{}, b1{}, b2{};
  bool available = false;
};

/// The frame of a primitive exact direction, built WHOLE ON THE PRODUCT
/// RUNG and narrowed once at the end, where the bound is a genuine
/// invariant instead of a gate across the middle of an algorithm.
///
/// The Lagrange loop always reduces the LONGER vector against the shorter:
/// the longer vector's own Gram term is the single quantity here that
/// outgrows the product rung, and reducing in this direction never forms it.
///
/// THE BEZOUT VECTOR IS LEFT AS THE RECURRENCE STATED IT, which is what the
/// narrow lane's own frame states, so the lift the two lanes begin from is
/// one value. It needs no shortening to fit: `r` is bounded by the band against
/// the direction, so `r * s` stands at most 443 bits on the widest lattice
/// this lane serves, and the placement reduces it on the product rung before
/// asking it to stand anywhere narrower.
template <typename Int>
auto make_exact_plane_frame(
    const std::array<typename exact_lane<Int>::coefficient_type, 3> &n)
    -> exact_plane_frame<Int> {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using product_type = typename exact_lane<Int>::product_type;

  exact_plane_frame<Int> frame;
  std::array<product_type, 3> s{}, b1{}, b2{};

  coefficient_type u(0), v(0);
  const coefficient_type g = tf::exact::door::ext_gcd(n[0], n[1], u, v);
  if (g == coefficient_type(0)) {
    if (n[2] == coefficient_type(0))
      return frame;
    s = {product_type(0), product_type(0),
         n[2] > coefficient_type(0) ? product_type(1) : product_type(-1)};
    b1 = {product_type(1), product_type(0), product_type(0)};
    b2 = {product_type(0), product_type(1), product_type(0)};
  } else {
    coefficient_type a(0), b(0);
    tf::exact::door::ext_gcd(g, n[2], a, b);
    s = {product_type(a) * product_type(u), product_type(a) * product_type(v),
         product_type(b)};
    b1 = {-product_type(n[1] / g), product_type(n[0] / g), product_type(0)};
    b2 = {-(product_type(u) * product_type(n[2])),
          -(product_type(v) * product_type(n[2])), product_type(g)};
  }

  for (int step = 0; step < 24; ++step) {
    if (product_rung_screen(b2) < product_rung_screen(b1))
      std::swap(b1, b2);
    if (exact_dot<Int>(b1, b1) == product_type(0))
      break;
    std::array<product_type, 3> shortened = b2;
    reduce_on_product_rung<Int>(shortened, b1);
    if (product_rung_screen(shortened) >= product_rung_screen(b2))
      break;
    b2 = shortened;
  }

  const product_type bound(exact_placement_bound<Int>());
  const auto narrowed = [&bound](const std::array<product_type, 3> &from,
                                 std::array<coefficient_type, 3> &to) -> bool {
    for (std::size_t k = 0; k < 3; ++k) {
      if (from[k] > bound || from[k] < -bound)
        return false;
      to[k] = static_cast<coefficient_type>(from[k]);
    }
    return true;
  };
  if (!narrowed(s, frame.s) || !narrowed(b1, frame.b1) ||
      !narrowed(b2, frame.b2))
    return frame;
  frame.available = true;
  return frame;
}

} // namespace tf::exact::door::pool
