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

namespace tf::exact::offset_dot {

/// The sign of `n . h + r` certified in double precision, or zero when the
/// bound cannot decide. Each product carries two operands rounded once on
/// conversion and its own rounding (3 epsilon), the offset its conversion,
/// and the four-term sum 3 epsilon of its magnitude sum (6 epsilon). The
/// coefficient covers both with the second-order terms rounded up. The bound
/// holds where @ref tf::exact::float_filter_sound does.
template <typename N, typename H, typename R>
auto filter_sign(const N &n, const H &h, const R &r) -> int {
  const double nx = double(n[0]), ny = double(n[1]), nz = double(n[2]);
  const double hx = double(h[0]), hy = double(h[1]), hz = double(h[2]);
  const double offset = double(r);
  const double x = nx * hx, y = ny * hy, z = nz * hz;
  const double determinant = x + y + z + offset;
  const double permanent =
      std::abs(x) + std::abs(y) + std::abs(z) + std::abs(offset);
  constexpr double epsilon = 1.1102230246251565e-16;
  constexpr double coefficient = (6.0 + 64.0 * epsilon) * epsilon;
  const double error = coefficient * permanent;
  if (determinant > error)
    return 1;
  if (determinant < -error)
    return -1;
  return 0;
}

} // namespace tf::exact::offset_dot
