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

#include "../../core/config/verify.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tf::exact::bits {

/// A magnitude of `width` significant bits held in a `Bits`-bit type,
/// rounded to the nearest `Real` once, ties to even, as a correctly
/// rounded conversion of the whole value rounds it: the top 64 bits carry
/// every bit below them as one sticky bit, so their own conversion is the
/// only rounding, and the scale is an exact power of two. The shifts read
/// only bits inside the magnitude's width, so a signed `Wide` holding a
/// magnitude with its top bit set reads the same as an unsigned one. A
/// magnitude past `Real`'s range scales to infinity, which is what the
/// native conversion of such a value states.
///
/// The sticky bit stands below every bit `Real` keeps of the 64, and that
/// is what makes the one conversion the whole rounding; a significand of
/// 64 bits would hold the sticky bit itself and is refused.
///
/// `width` is the magnitude's own bit width, at most `Bits`.
template <typename Real, unsigned Bits, typename Wide>
auto rounded_real(const Wide &magnitude, unsigned width) -> Real {
  static_assert(std::is_floating_point_v<Real>,
                "a magnitude rounds into a floating-point type");
  static_assert(std::numeric_limits<Real>::digits < 64,
                "the sticky bit stands below the significand");
  if constexpr (tf::core::verification)
    tf::core::verify(width <= Bits &&
                         (width == Bits || (magnitude >> width) == Wide(0)) &&
                         (width == 0 || (magnitude >> (width - 1)) != Wide(0)),
                     "the stated width is the magnitude's own");
  if (width <= 64)
    return Real(static_cast<std::uint64_t>(magnitude));
  const unsigned shift = width - 64;
  const auto top = static_cast<std::uint64_t>(magnitude >> shift);
  const bool sticky = (magnitude << (Bits - shift)) != Wide(0);
  return std::ldexp(Real(top | (sticky ? 1u : 0u)), int(shift));
}

} // namespace tf::exact::bits
