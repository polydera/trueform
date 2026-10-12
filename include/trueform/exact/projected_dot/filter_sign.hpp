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
#include <cmath>

namespace tf::exact::projected_dot {

/// The sign of `(a . b)(l . l) - (a . l)(b . l)` certified in double
/// precision, or zero when the bound cannot decide. Every term multiplies
/// four operands, each rounded once on conversion (4 epsilon); each
/// three-term dot is formed within 3 epsilon of its magnitude sum, the two
/// products of dots add one rounding and the difference one more (8
/// epsilon). The coefficient covers the sum with the second-order terms
/// rounded up. The bound holds where @ref tf::exact::float_filter_sound does.
template <typename A, typename B, typename L>
auto filter_sign(const A &a, const B &b, const L &l) -> int {
  const double ax = double(a[0]), ay = double(a[1]), az = double(a[2]);
  const double bx = double(b[0]), by = double(b[1]), bz = double(b[2]);
  const double lx = double(l[0]), ly = double(l[1]), lz = double(l[2]);
  const double ab = ax * bx + ay * by + az * bz;
  const double ll = lx * lx + ly * ly + lz * lz;
  const double al = ax * lx + ay * ly + az * lz;
  const double bl = bx * lx + by * ly + bz * lz;
  const double determinant = ab * ll - al * bl;
  const double permanent =
      (std::abs(ax * bx) + std::abs(ay * by) + std::abs(az * bz)) * ll +
      (std::abs(ax * lx) + std::abs(ay * ly) + std::abs(az * lz)) *
          (std::abs(bx * lx) + std::abs(by * ly) + std::abs(bz * lz));
  constexpr double epsilon = 1.1102230246251565e-16;
  constexpr double coefficient = (12.0 + 256.0 * epsilon) * epsilon;
  const double error = coefficient * permanent;
  if (determinant > error)
    return 1;
  if (determinant < -error)
    return -1;
  return 0;
}

} // namespace tf::exact::projected_dot
