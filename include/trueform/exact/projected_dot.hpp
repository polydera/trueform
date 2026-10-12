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
#include "./projected_dot/filter_sign.hpp"

namespace tf::exact {

/// The exact sign of `a . b` once `l` is projected out of both, scaled by
/// `l . l`: `(a . b)(l . l) - (a . l)(b . l)`, `-1`, `0` or `+1`, formed in
/// `Wide` — a rung that holds a product of four operand widths.
/// @ref tf::exact::projected_dot::filter_sign answers first wherever
/// @ref tf::exact::float_filter_sound holds; the exact value decides what it
/// leaves undecided.
template <typename Wide, typename A, typename B, typename L>
auto projected_dot_sign(const A &a, const B &b, const L &l) -> int {
  if constexpr (float_filter_sound) {
    const int filtered = projected_dot::filter_sign(a, b, l);
    if (filtered)
      return filtered;
  }
  const auto dot = [](const auto &u, const auto &v) {
    return Wide(u[0]) * Wide(v[0]) + Wide(u[1]) * Wide(v[1]) +
           Wide(u[2]) * Wide(v[2]);
  };
  const Wide value = dot(a, b) * dot(l, l) - dot(a, l) * dot(b, l);
  return value > Wide(0) ? 1 : value < Wide(0) ? -1 : 0;
}

} // namespace tf::exact
