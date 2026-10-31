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
#include "../difference_image.hpp"
#include "../meta.hpp"
#include "../vertex.hpp"

namespace tf::exact::incircle {

/// The incircle sign certified in double precision, or zero when the bound
/// cannot decide; Shewchuk's incircle A-bound,
/// https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c
///
/// The images are the coordinate differences of
/// @ref tf::exact::difference_image, exact below 2^52 and rounded to nearest
/// above it, which a 64-bit lattice reaches across a few dozen of its own
/// edges. From exact images the error is Shewchuk's (10 + 96 eps) eps times
/// the computed permanent. A rounded image carries relative error at most
/// eps, so each degree-four monomial of the determinant moves by at most
/// gamma_4 = 4 eps / (1 - 4 eps) of itself, and the determinant by gamma_4
/// times the permanent P of the exact differences. P is at most the
/// computed permanent over (1 - eps)^4 for the images and (1 - eps)^7 for
/// the at most seven roundings of a monomial's path through the computed
/// permanent, so the total error stays below (14 + 156 eps) eps times the
/// computed permanent plus terms of order eps^3, and (14 + 512 eps) eps is
/// the coefficient stated. A difference the native width cannot form is
/// refused. The bound holds where @ref tf::exact::float_filter_sound does.
template <typename Int, typename Coord>
auto filter_sign(const pt2<Coord> &a, const pt2<Coord> &b, const pt2<Coord> &c,
                 const pt2<Coord> &d) -> int {
  using T1 = typename meta<Int>::T1;
  double adx, ady, bdx, bdy, cdx, cdy;
  const difference_image_kind kinds[6] = {
      difference_image<T1>(a[0], d[0], adx),
      difference_image<T1>(a[1], d[1], ady),
      difference_image<T1>(b[0], d[0], bdx),
      difference_image<T1>(b[1], d[1], bdy),
      difference_image<T1>(c[0], d[0], cdx),
      difference_image<T1>(c[1], d[1], cdy)};
  bool rounded = false;
  for (const auto kind : kinds) {
    if (kind == difference_image_kind::refused)
      return 0;
    rounded = rounded || kind == difference_image_kind::rounded;
  }
  const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const double cdxady = cdx * ady, adxcdy = adx * cdy;
  const double adxbdy = adx * bdy, bdxady = bdx * ady;
  const double alift = adx * adx + ady * ady;
  const double blift = bdx * bdx + bdy * bdy;
  const double clift = cdx * cdx + cdy * cdy;
  const double determinant = alift * (bdxcdy - cdxbdy) +
                             blift * (cdxady - adxcdy) +
                             clift * (adxbdy - bdxady);
  const auto magnitude = [](double v) { return v < 0 ? -v : v; };
  const double permanent = (magnitude(bdxcdy) + magnitude(cdxbdy)) * alift +
                           (magnitude(cdxady) + magnitude(adxcdy)) * blift +
                           (magnitude(adxbdy) + magnitude(bdxady)) * clift;
  constexpr double epsilon = 1.1102230246251565e-16;
  constexpr double exact_coefficient = (10.0 + 96.0 * epsilon) * epsilon;
  constexpr double rounded_coefficient = (14.0 + 512.0 * epsilon) * epsilon;
  const double error =
      (rounded ? rounded_coefficient : exact_coefficient) * permanent;
  if (determinant > error)
    return 1;
  if (determinant < -error)
    return -1;
  return 0;
}

} // namespace tf::exact::incircle
