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

/// The non-negative greatest common divisor of two wide integers.
template <typename Wide> auto gcd_of(Wide a, Wide b) -> Wide {
  if (a < Wide(0))
    a = -a;
  if (b < Wide(0))
    b = -b;
  while (b != Wide(0)) {
    const Wide t = a % b;
    a = b;
    b = t;
  }
  return a;
}

} // namespace tf::exact::door
