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

#include "../meta.hpp"
#include "./gcd_of.hpp"
#include "./round_to_wide.hpp"

#include <array>
#include <cmath>

namespace tf::exact::door {

/// The direction on the door's grid: the unit direction scaled to
/// `steps`, rounded, reduced by the gcd of its components and turned so
/// the first nonzero one is positive. False when the direction is zero
/// or rounds onto it — a face that states no direction states no plane.
template <typename Int>
auto quantize_direction(double nx, double ny, double nz,
                        typename tf::exact::meta<Int>::T1 steps,
                        std::array<typename tf::exact::meta<Int>::T1, 3>
                            &normal) -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  const double length = std::sqrt(nx * nx + ny * ny + nz * nz);
  if (!(length > 0.0) || !std::isfinite(length))
    return false;
  const double scale = static_cast<double>(steps) / length;
  normal = {round_to_wide<Int>(nx * scale), round_to_wide<Int>(ny * scale),
            round_to_wide<Int>(nz * scale)};
  const T1 divisor = gcd_of(gcd_of(normal[0], normal[1]), normal[2]);
  if (divisor == T1(0))
    return false;
  normal = {normal[0] / divisor, normal[1] / divisor, normal[2] / divisor};
  if (normal[0] < T1(0) || (normal[0] == T1(0) && normal[1] < T1(0)) ||
      (normal[0] == T1(0) && normal[1] == T1(0) && normal[2] < T1(0)))
    normal = {-normal[0], -normal[1], -normal[2]};
  return true;
}

} // namespace tf::exact::door
