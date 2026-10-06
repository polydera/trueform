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
#include "../vertex.hpp"
#include <cmath>
#include <cstdint>

namespace tf::exact::insphere {

/// The insphere sign certified in double precision from exactly formed
/// integer differences, or zero when the bound cannot decide; Shewchuk's
/// insphere A-bound,
/// https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c
/// A difference is admitted below 2^52, which every 32-bit coordinate pair
/// meets by its width alone. The bound holds where
/// @ref tf::exact::float_filter_sound does.
template <typename Int, typename Coord>
auto filter_sign(const pt3<Coord> &a, const pt3<Coord> &b, const pt3<Coord> &c,
                 const pt3<Coord> &d, const pt3<Coord> &e) -> int {
  using T1 = typename meta<Int>::T1;
  const T1 axi = T1(a[0]) - e[0], ayi = T1(a[1]) - e[1], azi = T1(a[2]) - e[2];
  const T1 bxi = T1(b[0]) - e[0], byi = T1(b[1]) - e[1], bzi = T1(b[2]) - e[2];
  const T1 cxi = T1(c[0]) - e[0], cyi = T1(c[1]) - e[1], czi = T1(c[2]) - e[2];
  const T1 dxi = T1(d[0]) - e[0], dyi = T1(d[1]) - e[1], dzi = T1(d[2]) - e[2];
  if constexpr (sizeof(Coord) > sizeof(std::int32_t)) {
    const auto exact_double = [](T1 v) {
      const T1 limit = T1(1) << 52;
      return v > -limit && v < limit;
    };
    if (!(exact_double(axi) && exact_double(ayi) && exact_double(azi) &&
          exact_double(bxi) && exact_double(byi) && exact_double(bzi) &&
          exact_double(cxi) && exact_double(cyi) && exact_double(czi) &&
          exact_double(dxi) && exact_double(dyi) && exact_double(dzi)))
      return 0;
  }

  const double ax = double(axi), ay = double(ayi), az = double(azi);
  const double bx = double(bxi), by = double(byi), bz = double(bzi);
  const double cx = double(cxi), cy = double(cyi), cz = double(czi);
  const double dx = double(dxi), dy = double(dyi), dz = double(dzi);
  const double axby = ax * by, bxay = bx * ay;
  const double bxcy = bx * cy, cxby = cx * by;
  const double cxdy = cx * dy, dxcy = dx * cy;
  const double dxay = dx * ay, axdy = ax * dy;
  const double axcy = ax * cy, cxay = cx * ay;
  const double bxdy = bx * dy, dxby = dx * by;
  const double ab = axby - bxay, bc = bxcy - cxby;
  const double cd = cxdy - dxcy, da = dxay - axdy;
  const double ac = axcy - cxay, bd = bxdy - dxby;
  const double abc = az * bc - bz * ac + cz * ab;
  const double bcd = bz * cd - cz * bd + dz * bc;
  const double cda = cz * da + dz * ac + az * cd;
  const double dab = dz * ab + az * bd + bz * da;
  const double alift = ax * ax + ay * ay + az * az;
  const double blift = bx * bx + by * by + bz * bz;
  const double clift = cx * cx + cy * cy + cz * cz;
  const double dlift = dx * dx + dy * dy + dz * dz;
  const double determinant =
      (dlift * abc - clift * dab) + (blift * cda - alift * bcd);
  const double permanent = ((std::abs(cxdy) + std::abs(dxcy)) * std::abs(bz) +
                            (std::abs(dxby) + std::abs(bxdy)) * std::abs(cz) +
                            (std::abs(bxcy) + std::abs(cxby)) * std::abs(dz)) *
                               alift +
                           ((std::abs(dxay) + std::abs(axdy)) * std::abs(cz) +
                            (std::abs(axcy) + std::abs(cxay)) * std::abs(dz) +
                            (std::abs(cxdy) + std::abs(dxcy)) * std::abs(az)) *
                               blift +
                           ((std::abs(axby) + std::abs(bxay)) * std::abs(dz) +
                            (std::abs(bxdy) + std::abs(dxby)) * std::abs(az) +
                            (std::abs(dxay) + std::abs(axdy)) * std::abs(bz)) *
                               clift +
                           ((std::abs(bxcy) + std::abs(cxby)) * std::abs(az) +
                            (std::abs(cxay) + std::abs(axcy)) * std::abs(bz) +
                            (std::abs(axby) + std::abs(bxay)) * std::abs(cz)) *
                               dlift;
  constexpr double epsilon = 1.1102230246251565e-16;
  constexpr double coefficient = (16.0 + 224.0 * epsilon) * epsilon;
  const double error = coefficient * permanent;
  if (determinant > error)
    return -1;
  if (determinant < -error)
    return 1;
  return 0;
}

} // namespace tf::exact::insphere
