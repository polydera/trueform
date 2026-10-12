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

#include "./float_filter_sound.hpp"
#include "./offset_dot/filter_sign.hpp"

namespace tf::exact {

/// The exact sign of `n . h + r`, `-1`, `0` or `+1`, formed in `Wide` — a
/// rung that holds a product of the two vector operands' widths and the
/// offset. @ref tf::exact::offset_dot::filter_sign answers first wherever
/// @ref tf::exact::float_filter_sound holds; the exact value decides what it
/// leaves undecided.
template <typename Wide, typename N, typename H, typename R>
auto offset_dot_sign(const N &n, const H &h, const R &r) -> int {
  if constexpr (float_filter_sound) {
    const int filtered = offset_dot::filter_sign(n, h, r);
    if (filtered)
      return filtered;
  }
  const Wide value = Wide(n[0]) * Wide(h[0]) + Wide(n[1]) * Wide(h[1]) +
                     Wide(n[2]) * Wide(h[2]) + Wide(r);
  return value > Wide(0) ? 1 : value < Wide(0) ? -1 : 0;
}

} // namespace tf::exact
