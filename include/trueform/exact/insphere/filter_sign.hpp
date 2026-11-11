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
#include "../../core/config/inline.hpp"
#include "../difference_image.hpp"
#include "../vertex.hpp"
#include <algorithm>
#include <cmath>

namespace tf::exact::insphere {

/// The insphere sign certified in double precision, or zero when the bound
/// cannot decide.
///
/// The images are the coordinate differences of
/// @ref tf::exact::difference_image, exact below 2^52 and rounded to nearest
/// above it, which a 64-bit lattice reaches across a few dozen of its own
/// edges. From exact images the error is (16 + 224 eps) eps times the
/// computed permanent. A rounded image carries relative error at most eps,
/// so each degree-five monomial of the determinant moves by at most
/// gamma_5 = 5 eps / (1 - 5 eps) of itself, and the determinant by gamma_5
/// times the permanent P of the exact differences. P is at most the
/// computed permanent over (1 - eps)^5 for the images and (1 - eps)^9 for
/// the at most nine roundings of a monomial's path through the computed
/// permanent, so the total error stays below (21 + 400 eps) eps times the
/// computed permanent plus terms of order eps^3, and (21 + 1024 eps) eps is
/// the coefficient stated. A difference the coordinate cannot hold is
/// refused. The bound holds where @ref tf::exact::float_filter_sound does.
/// The permanent is asked only of a determinant its cover does not decide:
/// each of its four cofactors is six products of one x, one y and one z
/// image, so it is at most six times the largest image on each axis times
/// the sum of the lifts, and the cover's margin of 2^-40 is past the
/// seventeen roundings between the two, so a determinant beyond the cover
/// is beyond the bound.
/// `Exact` is the caller's fact that every coordinate difference of its
/// storage is exact in a double (a stored width of at most 52 bits), so the
/// images are formed as plain differences and nothing is classified.
template <bool Exact = false, typename Coord>
TF_FORCE_INLINE auto filter_sign(const pt3<Coord> &a, const pt3<Coord> &b,
                                 const pt3<Coord> &c, const pt3<Coord> &d,
                                 const pt3<Coord> &e) -> int {
  double ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz;
  bool rounded = false;
  if constexpr (Exact) {
    ax = double(a[0]) - double(e[0]);
    ay = double(a[1]) - double(e[1]);
    az = double(a[2]) - double(e[2]);
    bx = double(b[0]) - double(e[0]);
    by = double(b[1]) - double(e[1]);
    bz = double(b[2]) - double(e[2]);
    cx = double(c[0]) - double(e[0]);
    cy = double(c[1]) - double(e[1]);
    cz = double(c[2]) - double(e[2]);
    dx = double(d[0]) - double(e[0]);
    dy = double(d[1]) - double(e[1]);
    dz = double(d[2]) - double(e[2]);
  } else {
    const difference_image_kind kinds[12] = {
        difference_image(a[0], e[0], ax), difference_image(a[1], e[1], ay),
        difference_image(a[2], e[2], az), difference_image(b[0], e[0], bx),
        difference_image(b[1], e[1], by), difference_image(b[2], e[2], bz),
        difference_image(c[0], e[0], cx), difference_image(c[1], e[1], cy),
        difference_image(c[2], e[2], cz), difference_image(d[0], e[0], dx),
        difference_image(d[1], e[1], dy), difference_image(d[2], e[2], dz)};
    for (const auto kind : kinds) {
      if (kind == difference_image_kind::refused)
        return 0;
      rounded = rounded || kind == difference_image_kind::rounded;
    }
  }
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
  constexpr double epsilon = 1.1102230246251565e-16;
  constexpr double exact_coefficient = (16.0 + 224.0 * epsilon) * epsilon;
  constexpr double rounded_coefficient = (21.0 + 1024.0 * epsilon) * epsilon;
  const double coefficient = rounded ? rounded_coefficient : exact_coefficient;
  constexpr double reach = 6.0 * (1.0 + 0x1p-40);
  const double widest_x = std::max(std::max(std::abs(ax), std::abs(bx)),
                                   std::max(std::abs(cx), std::abs(dx)));
  const double widest_y = std::max(std::max(std::abs(ay), std::abs(by)),
                                   std::max(std::abs(cy), std::abs(dy)));
  const double widest_z = std::max(std::max(std::abs(az), std::abs(bz)),
                                   std::max(std::abs(cz), std::abs(dz)));
  const double cover = reach * coefficient * (widest_x * widest_y * widest_z) *
                       ((alift + blift) + (clift + dlift));
  if (std::abs(determinant) > cover)
    return determinant > 0.0 ? -1 : 1;
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
  const double error = coefficient * permanent;
  if (determinant > error)
    return -1;
  if (determinant < -error)
    return 1;
  return 0;
}

} // namespace tf::exact::insphere
