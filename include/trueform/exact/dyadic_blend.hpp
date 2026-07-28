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

/// @ingroup exact
/// @brief One coordinate of the dyadic blend of `a` and `b` at the
///        parameter, rounded to nearest.
///
/// The single authority for turning a parameter on an edge into a lattice
/// coordinate. Every producer of a split point goes through here: two
/// implementations of this rule that drift apart place the same identity
/// at two different coordinates, and a seam opens between the carriers
/// that disagree.
///
/// The parameter's width is a property of the lattice type, taken from
/// `meta<Int>`, not chosen per call site.
template <typename Int>
auto dyadic_blend(Int a, Int b,
                  typename tf::exact::meta<Int>::param_type parameter,
                  int bits = tf::exact::meta<Int>::param_bits) -> Int {
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const T2 a_weight = T2((param_t(1) << bits) - parameter);
  const T2 b_weight = T2(parameter);
  const T2 value = T2(a) * a_weight + T2(b) * b_weight;
  const T2 half = T2(param_t(1) << (bits - 1));
  return value < T2(0) ? -static_cast<Int>((-value + half) >> bits)
                       : static_cast<Int>((value + half) >> bits);
}

} // namespace tf::exact
