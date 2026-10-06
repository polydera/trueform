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
#include "./dyadic_blend_scaled.hpp"
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
  return dyadic_blend_scaled<Int>(a, b, parameter, bits);
}

} // namespace tf::exact
