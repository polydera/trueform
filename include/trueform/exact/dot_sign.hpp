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

#include "./bits/twos_complement.hpp"

#include <array>

namespace tf::exact {

/// Exact sign of a . b for 3-vectors whose component products overflow
/// the component type `T`.
///
/// Each product `a[c] * b[c]` is formed as two in-range partials
/// (`a` split at half the width of `T`), and the accumulator is
/// renormalized per term: the low limb stays in `[0, 2^h)`, its carry
/// moves up immediately, and the high limb carries a single
/// two's-complement extension bit — the sum of three high partials
/// exceeds the type by less than a factor of two, so one bit of
/// wrap-tracking makes the accumulation exact.
///
/// Valid while |a[c]| < 2^(2h-1) (the full range of `T`) and
/// |b[c]| < 2^(h-1), for h = half the bit width of `T` — covering the
/// arrangement predicates' degree-4 carriers against degree-2 wedge
/// operands at their attainable extremes.
template <typename T>
auto dot_sign(const std::array<T, 3> &a, const std::array<T, 3> &b) -> int {
  const unsigned h = unsigned(sizeof(T)) * 4u;
  T ovf = T(0);  // extension bit of s_hi, in units of 2^(2h)
  T s_hi = T(0); // scale 2^h
  T s_lo = T(0); // scale 1, kept in [0, 2^h)
  for (int c = 0; c < 3; ++c) {
    const T hi = bits::shift_right(a[c], h);
    const T rem = a[c] - bits::shift_left(hi, h); // [0, 2^h)
    const T p = rem * b[c];                       // |p| < 2^(2h-1), in range
    const T pc = bits::shift_right(p, h);
    s_lo = s_lo + (p - bits::shift_left(pc, h));
    const T cl = bits::shift_right(s_lo, h);
    s_lo = s_lo - bits::shift_left(cl, h);
    const T t = hi * b[c] + pc + cl; // |t| < 2^(2h-2) + small, in range
    const T prev = s_hi;
    s_hi = bits::wrapping_add(s_hi, t);
    if ((prev < T(0)) == (t < T(0)) && (s_hi < T(0)) != (prev < T(0)))
      ovf = ovf + (prev < T(0) ? T(-1) : T(1));
  }
  if (ovf != T(0))
    return ovf > T(0) ? 1 : -1;
  if (s_hi > T(0))
    return 1;
  if (s_hi < T(0))
    return -1;
  return (s_lo > T(0)) ? 1 : 0;
}

} // namespace tf::exact
