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

namespace tf::exact::door {

/// `num / den` rounded half away from zero, in integers. `den != 0`.
///
/// The tie is decided on the remainder, which lies inside `den`, so a
/// numerator anywhere in the type is answered without forming a multiple
/// of it.
template <typename Wide> auto round_div(Wide num, Wide den) -> Wide {
  if (den < Wide(0)) {
    den = -den;
    num = -num;
  }
  const Wide quotient = num / den;
  const Wide remainder = num - quotient * den;
  if (remainder >= Wide(0))
    return remainder >= den - remainder ? quotient + Wide(1) : quotient;
  const Wide magnitude = -remainder;
  return magnitude >= den - magnitude ? quotient - Wide(1) : quotient;
}

} // namespace tf::exact::door
