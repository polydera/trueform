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

#include <cmath>
#include <cstdint>

namespace tf::exact::door {

/// The bound every double-valued step of the placement is clamped to
/// before it re-enters integer arithmetic. It is half the reach of the
/// rung the placement is solved on, so a clamped value still multiplies
/// against a name and an offset on the rung ABOVE that one — which is
/// where @ref tf::exact::door::wide_dot, @ref tf::exact::door::wide_det3
/// and @ref tf::exact::door::wide_axpy form their products. It states
/// nothing about a product taken at the placement's own rung.
///
/// The bound is a fact of the lattice and not of the wide type alone: a
/// lattice twice as wide states a rung twice as wide and a bound twice
/// as far.
template <typename Int>
auto wide_placement_bound() -> typename tf::exact::meta<Int>::T1 {
  using T1 = typename tf::exact::meta<Int>::T1;
  return T1(1) << unsigned(tf::exact::meta<Int>::t1_bits - 1);
}

/// A double rounded into the rung above the lattice, clamped to
/// @ref tf::exact::door::wide_placement_bound. A candidate that
/// needs the clamp is outside the lattice the certificate can admit, so
/// clamping decides nothing the certificate does not then refuse.
///
/// A double past 2^53 is already an integer, and its mantissa and
/// exponent state it exactly at any width — so the value is carried at
/// the rung's own width and never through a 64-bit rounding.
template <typename Int>
auto round_to_wide(double v) -> typename tf::exact::meta<Int>::T1 {
  using T1 = typename tf::exact::meta<Int>::T1;

  const T1 bound = wide_placement_bound<Int>();
  const double limit = static_cast<double>(bound);
  if (!(v > -limit))
    return -bound;
  if (!(v < limit))
    return bound;
  const double rounded = std::round(v);
  int exponent = 0;
  const double fraction = std::frexp(rounded, &exponent);
  if (exponent <= 53)
    return T1(std::llround(rounded));
  const bool negative = fraction < 0.0;
  const auto mantissa = static_cast<std::int64_t>(
      std::ldexp(negative ? -fraction : fraction, 53));
  const T1 magnitude = T1(mantissa) << unsigned(exponent - 53);
  return negative ? -magnitude : magnitude;
}

} // namespace tf::exact::door
