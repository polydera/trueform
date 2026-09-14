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
#include "./exact_placement_bound.hpp"

#include <cmath>
#include <cstdint>

namespace tf::exact::door::pool {

/// A double rounded onto the coefficient rung, clamped to
/// @ref tf::exact::door::pool::exact_placement_bound. A double past 2^53 is
/// already an integer and its mantissa and exponent state it exactly at
/// any width, so the value is carried at the rung's own width and never
/// through a 64-bit rounding.
template <typename Int>
auto round_to_exact(double v) -> typename exact_lane<Int>::coefficient_type {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;

  const coefficient_type bound = exact_placement_bound<Int>();
  const double limit = static_cast<double>(bound);
  if (!(v > -limit))
    return -bound;
  if (!(v < limit))
    return bound;
  const double rounded = std::round(v);
  int exponent = 0;
  const double fraction = std::frexp(rounded, &exponent);
  if (exponent <= 53)
    return coefficient_type(std::llround(rounded));
  const bool negative = fraction < 0.0;
  const auto mantissa = static_cast<std::int64_t>(
      std::ldexp(negative ? -fraction : fraction, 53));
  const coefficient_type magnitude = coefficient_type(mantissa)
                                     << unsigned(exponent - 53);
  return negative ? -magnitude : magnitude;
}

} // namespace tf::exact::door::pool
