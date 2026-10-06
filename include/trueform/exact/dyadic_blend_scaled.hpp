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
#include "./meta.hpp"

namespace tf::exact {

/// Dyadic blend on a scaled lattice, rounded to nearest. The weighted
/// coordinates and rounding offset must fit meta<Int>::T2.
template <typename Int, typename Coord>
auto dyadic_blend_scaled(Coord a, Coord b,
                         typename meta<Int>::param_type parameter,
                         int bits = meta<Int>::param_bits) -> Coord {
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const T2 a_weight = T2((param_t(1) << bits) - parameter);
  const T2 b_weight = T2(parameter);
  const T2 value = T2(a) * a_weight + T2(b) * b_weight;
  const T2 half = T2(param_t(1) << (bits - 1));
  return value < T2(0) ? -static_cast<Coord>((-value + half) >> bits)
                       : static_cast<Coord>((value + half) >> bits);
}

} // namespace tf::exact
