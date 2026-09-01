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
/// @brief @ref tf::exact::dyadic_ratio for fractions past its width
///        budget: same value, same rounding, computed by chunked
///        division so `numerator * 2^bits` is never materialized.
///
/// dyadic_ratio multiplies the numerator by the scale inside T2, which
/// admits only its stated degree-2 budget; an edge–plane fraction is
/// degree 3 and overflows it. Here the quotient's bits are produced in
/// chunks sized from the denominator's actual headroom — every
/// intermediate `remainder << chunk` stays in T2, and a ladder
/// denominator's margin makes the loop two iterations, each one fused
/// division (the remainder is recovered by multiply-subtract).
template <typename Int>
auto wide_dyadic_ratio(typename tf::exact::meta<Int>::T2 numerator,
                       typename tf::exact::meta<Int>::T2 denominator,
                       int bits = tf::exact::meta<Int>::param_bits) ->
    typename tf::exact::meta<Int>::param_type {
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  if (denominator <= T2(0) || numerator <= T2(0))
    return param_t(0);
  if (denominator <= numerator)
    return param_t(param_t(1) << bits);
  constexpr int t2_bits = int(sizeof(T2) * 8);
  int den_width = 1;
  {
    T2 x = denominator;
    for (int s = t2_bits / 2; s; s >>= 1) {
      const T2 y = x >> unsigned(s);
      if (!(y == T2(0))) {
        x = y;
        den_width += s;
      }
    }
  }
  const int chunk = t2_bits - 1 - den_width;
  T2 quotient(0);
  T2 remainder = numerator;
  for (int produced = 0; produced < bits;) {
    const int c = bits - produced < chunk ? bits - produced : chunk;
    remainder = remainder << unsigned(c);
    const T2 digit = remainder / denominator;
    remainder = remainder - digit * denominator;
    quotient = (quotient << unsigned(c)) + digit;
    produced += c;
  }
  if (remainder + remainder >= denominator)
    quotient = quotient + T2(1);
  return param_t(quotient);
}

} // namespace tf::exact
