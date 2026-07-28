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
/// @brief `numerator / denominator` as a dyadic parameter, rounded to
///        nearest and clamped to `[0, 2^param_bits]`.
///
/// The inverse of @ref tf::exact::dyadic_blend and the single authority
/// for turning a position along an edge into a transportable parameter.
/// A caller that requires the parameter to be strictly interior tests the
/// result against the endpoints rather than reimplementing the rounding.
///
/// The denominator is assumed positive; pass the squared edge length, or
/// a cross-product denominator already normalised by its sign.
template <typename Int>
auto dyadic_ratio(typename tf::exact::meta<Int>::T2 numerator,
                  typename tf::exact::meta<Int>::T2 denominator,
                  int bits = tf::exact::meta<Int>::param_bits) ->
    typename tf::exact::meta<Int>::param_type {
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const T2 scale = T2(param_t(1) << bits);
  if (denominator <= T2(0) || numerator <= T2(0))
    return param_t(0);
  if (denominator <= numerator)
    return param_t(scale);
  return param_t((numerator * scale + denominator / T2(2)) / denominator);
}

} // namespace tf::exact
