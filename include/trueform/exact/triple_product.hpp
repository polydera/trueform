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
#include "./triple_product/filter_sign.hpp"

namespace tf::exact {

/// The exact sign of `(a x b) . c`, `-1`, `0` or `+1`, the expansion formed
/// in `Wide` — a rung that holds the product of the three operands' widths.
/// @ref tf::exact::triple_product::filter_sign answers first wherever
/// @ref tf::exact::float_filter_sound holds; the exact value decides what it
/// leaves undecided.
template <typename Wide, typename A, typename B, typename C>
auto triple_product_sign(const A &a, const B &b, const C &c) -> int {
  if constexpr (float_filter_sound) {
    const int filtered = triple_product::filter_sign(a, b, c);
    if (filtered)
      return filtered;
  }
  const Wide value =
      (Wide(a[1]) * Wide(b[2]) - Wide(a[2]) * Wide(b[1])) * Wide(c[0]) +
      (Wide(a[2]) * Wide(b[0]) - Wide(a[0]) * Wide(b[2])) * Wide(c[1]) +
      (Wide(a[0]) * Wide(b[1]) - Wide(a[1]) * Wide(b[0])) * Wide(c[2]);
  return value > Wide(0) ? 1 : value < Wide(0) ? -1 : 0;
}

} // namespace tf::exact
