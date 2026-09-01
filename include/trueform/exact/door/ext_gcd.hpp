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

/// Bezout: the gcd of `a` and `b`, with `x` and `y` solving
/// `a x + b y = gcd`. The gcd is returned non-negative, so the pair is
/// the one belonging to that representative.
template <typename Wide>
auto ext_gcd(Wide a, Wide b, Wide &x, Wide &y) -> Wide {
  Wide old_r = a, r = b;
  Wide old_s = Wide(1), s = Wide(0);
  Wide old_t = Wide(0), t = Wide(1);
  while (r != Wide(0)) {
    const Wide q = old_r / r;
    Wide next = old_r - q * r;
    old_r = r;
    r = next;
    next = old_s - q * s;
    old_s = s;
    s = next;
    next = old_t - q * t;
    old_t = t;
    t = next;
  }
  if (old_r < Wide(0)) {
    old_r = -old_r;
    old_s = -old_s;
    old_t = -old_t;
  }
  x = old_s;
  y = old_t;
  return old_r;
}

} // namespace tf::exact::door
