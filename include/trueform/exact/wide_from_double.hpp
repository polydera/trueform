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
#include <cmath>
#include <cstdint>

namespace tf::exact {

/// The integer part of a finite `v` as `Wide`, formed as the native cast
/// forms it: truncated toward zero, and past 2^63 carried as its mantissa
/// shifted to its exponent, so every double that is an integer converts
/// exactly at any width. A value past the width is the caller's error, as
/// it is for the native cast.
template <typename Wide> auto wide_from_double(double v) noexcept -> Wide {
  const double truncated = std::trunc(v);
  int exponent = 0;
  const double fraction = std::frexp(truncated, &exponent);
  if (exponent <= 63)
    return Wide(static_cast<std::int64_t>(truncated));
  const bool negative = fraction < 0.0;
  const auto mantissa = static_cast<std::int64_t>(
      std::ldexp(negative ? -fraction : fraction, 53));
  const Wide magnitude = Wide(mantissa) << unsigned(exponent - 53);
  return negative ? -magnitude : magnitude;
}

} // namespace tf::exact
