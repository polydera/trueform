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

namespace tf::exact::triple_product {

/// The sign of `(a x b) . c` certified in double precision, or zero when the
/// bound cannot decide. The operands are integers of any width, each rounded
/// once on conversion (relative error at most epsilon), which moves every
/// product of three by at most 3 epsilon of its magnitude; the expansion's
/// own evaluation is Shewchuk's orient3d A-bound, 7 epsilon,
/// https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c
/// The coefficient covers both with the second-order terms rounded up. The
/// bound holds where @ref tf::exact::float_filter_sound does.
template <typename A, typename B, typename C>
auto filter_sign(const A &a, const B &b, const C &c) -> int {
  const double ax = double(a[0]), ay = double(a[1]), az = double(a[2]);
  const double bx = double(b[0]), by = double(b[1]), bz = double(b[2]);
  const double cx = double(c[0]), cy = double(c[1]), cz = double(c[2]);
  const double aybz = ay * bz, azby = az * by;
  const double azbx = az * bx, axbz = ax * bz;
  const double axby = ax * by, aybx = ay * bx;
  const double determinant =
      (aybz - azby) * cx + (azbx - axbz) * cy + (axby - aybx) * cz;
  const double permanent = (std::abs(aybz) + std::abs(azby)) * std::abs(cx) +
                           (std::abs(azbx) + std::abs(axbz)) * std::abs(cy) +
                           (std::abs(axby) + std::abs(aybx)) * std::abs(cz);
  constexpr double epsilon = 1.1102230246251565e-16;
  constexpr double coefficient = (10.0 + 128.0 * epsilon) * epsilon;
  const double error = coefficient * permanent;
  if (determinant > error)
    return 1;
  if (determinant < -error)
    return -1;
  return 0;
}

} // namespace tf::exact::triple_product
