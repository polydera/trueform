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

namespace tf::exact::door::pool {

/// THE NARROW RUNG'S ADMISSION WIDTH. Three products of two values this
/// wide still stand on the coefficient rung: `2k + 2 <= bound_bits + 1`.
template <typename Int>
inline constexpr int narrow_product_bits = (exact_lane<Int>::bound_bits - 1) / 2;

/// Whether a value is narrow enough for the coefficient rung to carry its
/// products. Asked of the ACTUAL operands, never of their declared type.
template <typename Int, typename Value>
auto stands_narrow(const Value &v) -> bool {
  const Value bound = Value(1) << unsigned(narrow_product_bits<Int>);
  return v < bound && v > -bound;
}

/// The dot product of two coefficient triples, on the product rung.
///
/// TWO LANES, ONE ANSWER, CHOSEN BY WIDTH. A normal is a cross of coordinate
/// differences, so a mesh whose faces are small against its own extent
/// states every direction a rung below what the product type has to allow
/// for. Where all six components admit it the whole dot is formed on the
/// COEFFICIENT rung and widened once. Both lanes are exact, so the integer
/// is the same and the lane moves the price alone.
template <typename Int, typename A, typename B>
auto exact_dot(const A &a, const B &b) ->
    typename exact_lane<Int>::product_type {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using product_type = typename exact_lane<Int>::product_type;
  if (stands_narrow<Int>(a[0]) && stands_narrow<Int>(a[1]) &&
      stands_narrow<Int>(a[2]) && stands_narrow<Int>(b[0]) &&
      stands_narrow<Int>(b[1]) && stands_narrow<Int>(b[2]))
    return product_type(coefficient_type(a[0]) * coefficient_type(b[0]) +
                        coefficient_type(a[1]) * coefficient_type(b[1]) +
                        coefficient_type(a[2]) * coefficient_type(b[2]));
  return product_type(a[0]) * product_type(b[0]) +
         product_type(a[1]) * product_type(b[1]) +
         product_type(a[2]) * product_type(b[2]);
}

/// `a * b` on the narrow rung where both admit it, on the product rung
/// where they do not. The same integer either way.
template <typename Int>
auto exact_product(const typename exact_lane<Int>::product_type &a,
                   const typename exact_lane<Int>::product_type &b) ->
    typename exact_lane<Int>::product_type {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using product_type = typename exact_lane<Int>::product_type;
  if (stands_narrow<Int>(a) && stands_narrow<Int>(b))
    return product_type(coefficient_type(a) * coefficient_type(b));
  return a * b;
}

} // namespace tf::exact::door::pool
