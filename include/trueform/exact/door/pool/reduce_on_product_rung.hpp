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
#include "./wide_to_double.hpp"

#include <array>
#include <cstddef>

namespace tf::exact::door::pool {

/// The squared length of a triple as a double: the SCREEN a reduction
/// orders its operands by.
///
/// A misjudgement costs a worse basis and never a wrong lattice —
/// subtracting an integer multiple of one basis vector from the other is
/// unimodular whatever the multiple is. Ordering by a screen keeps THE
/// LONGER VECTOR'S OWN GRAM TERM out of the arithmetic, that being the one
/// quantity of this construction that does not stand on the product rung.
template <typename V> auto product_rung_screen(const V &v) -> double {
  const double x = wide_to_double(v[0]);
  const double y = wide_to_double(v[1]);
  const double z = wide_to_double(v[2]);
  return x * x + y * y + z * z;
}

/// `numerator / denominator` rounded to nearest, halves away from zero.
///
/// The half is decided as `rem >= span - rem` rather than by doubling the
/// remainder, because a remainder of a product-rung denominator has no room
/// above it to be doubled in.
template <typename Product>
auto product_rung_quotient(const Product &numerator,
                           const Product &denominator) -> Product {
  auto [quotient, remainder] = divmod(numerator, denominator);
  if (remainder < Product(0))
    remainder = -remainder;
  Product span = denominator;
  if (span < Product(0))
    span = -span;
  if (remainder >= span - remainder)
    quotient = quotient + ((numerator < Product(0)) != (denominator < Product(0))
                               ? Product(-1)
                               : Product(1));
  return quotient;
}

/// One Babai pass: `at` moved by the multiple of `b` that brings it nearest
/// the line `b` spans.
///
/// `at` stands on the product rung and may sit above the coefficient rung —
/// bringing it back down is what the pass is for. `b` may be stated at
/// either rung. The only Gram term formed is `b . b`, so the caller governs
/// the width of the whole pass by choosing which vector it hands as `b`.
template <typename Int, typename B>
auto reduce_on_product_rung(
    std::array<typename exact_lane<Int>::product_type, 3> &at, const B &b)
    -> void {
  using product_type = typename exact_lane<Int>::product_type;
  const product_type square = exact_dot<Int>(b, b);
  if (square == product_type(0))
    return;
  const product_type step =
      product_rung_quotient(exact_dot<Int>(at, b), square);
  for (std::size_t k = 0; k < 3; ++k)
    at[k] = at[k] - step * product_type(b[k]);
}

} // namespace tf::exact::door::pool
