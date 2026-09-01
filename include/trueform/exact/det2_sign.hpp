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
#include "./meta.hpp"

namespace tf::exact {

/// Exact sign of a*b - c*d for operands on the T2 rung of `Int`'s
/// ladder — the 2x2 determinant sign, and with it the order of two
/// exact rationals: sign(n1/d1 - n2/d2) = det2_sign<Int>(n1, d2, n2, d1)
/// once the denominator signs are normalized. The product difference
/// spans two T2 widths; no wider rung exists, so the sign is carried
/// in limbs.
///
/// Each operand splits into three limbs at a third of the T2 width
/// (low limbs in [0, 2^h), top limb signed), so every partial product
/// fits T2 with room to accumulate: the nine partials per product land
/// on five scale accumulators, a single low-to-high carry pass
/// renormalizes the low four into [0, 2^h), and the top accumulator
/// alone carries the sign. Total over T2: any in-range operands, no
/// operand contract.
template <typename Int>
auto det2_sign(const typename meta<Int>::T2 &a, const typename meta<Int>::T2 &b,
               const typename meta<Int>::T2 &c, const typename meta<Int>::T2 &d)
    -> int {
  using T2 = typename meta<Int>::T2;
  const unsigned h = unsigned(meta<Int>::t2_bits + 1) / 3u;
  const auto split = [](const T2 &x, unsigned w, T2(&out)[3]) {
    const T2 x2 = bits::shift_right(x, 2u * w);
    const T2 r = x - bits::shift_left(x2, 2u * w);
    const T2 x1 = bits::shift_right(r, w);
    out[0] = r - bits::shift_left(x1, w);
    out[1] = x1;
    out[2] = x2;
  };
  T2 A[3], B[3], C[3], D[3];
  split(a, h, A);
  split(b, h, B);
  split(c, h, C);
  split(d, h, D);
  T2 s[5] = {T2(0), T2(0), T2(0), T2(0), T2(0)};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      s[i + j] = s[i + j] + A[i] * B[j] - C[i] * D[j];
  for (int k = 0; k < 4; ++k) {
    const T2 carry = bits::shift_right(s[k], h);
    s[k] = s[k] - bits::shift_left(carry, h);
    s[k + 1] = s[k + 1] + carry;
  }
  if (s[4] != T2(0))
    return s[4] > T2(0) ? 1 : -1;
  for (int k = 3; k >= 0; --k)
    if (s[k] != T2(0))
      return 1;
  return 0;
}

} // namespace tf::exact
