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
#include "./difference_image.hpp"
#include "./meta.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// The orient2d sign certified in double precision, or zero when the bound
/// cannot decide; Shewchuk's orient2d A-bound,
/// https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c
///
/// The images are the coordinate differences of
/// @ref tf::exact::difference_image. From exact images the error is
/// Shewchuk's (3 + 16 eps) eps times the computed permanent. A rounded image
/// carries relative error at most eps, so each degree-two product moves by
/// at most gamma_2 = 2 eps / (1 - 2 eps) of itself, and the determinant by
/// gamma_2 times the permanent of the exact differences; that permanent is
/// at most the computed one over (1 - eps)^2 for the images and (1 - eps)^2
/// for its own two roundings, so the total stays below (5 + 28 eps) eps
/// times the computed permanent plus terms of order eps^3, and
/// (5 + 64 eps) eps is the coefficient stated. A difference the native width
/// cannot form is refused. The bound holds where
/// @ref tf::exact::float_filter_sound does.
template <typename Int, typename Coord>
auto orient2d_filter_sign(const pt2<Coord> &a, const pt2<Coord> &b,
                          const pt2<Coord> &c) -> int {
  using T1 = typename meta<Int>::T1;
  double bax, bay, cax, cay;
  const difference_image_kind kinds[4] = {
      difference_image<T1>(b[0], a[0], bax),
      difference_image<T1>(b[1], a[1], bay),
      difference_image<T1>(c[0], a[0], cax),
      difference_image<T1>(c[1], a[1], cay)};
  bool rounded = false;
  for (const auto kind : kinds) {
    if (kind == difference_image_kind::refused)
      return 0;
    rounded = rounded || kind == difference_image_kind::rounded;
  }
  const double left = bax * cay;
  const double right = bay * cax;
  const double determinant = left - right;
  const auto magnitude = [](double v) { return v < 0 ? -v : v; };
  const double permanent = magnitude(left) + magnitude(right);
  constexpr double epsilon = 1.1102230246251565e-16;
  constexpr double exact_coefficient = (3.0 + 16.0 * epsilon) * epsilon;
  constexpr double rounded_coefficient = (5.0 + 64.0 * epsilon) * epsilon;
  const double error =
      (rounded ? rounded_coefficient : exact_coefficient) * permanent;
  if (determinant > error)
    return 1;
  if (determinant < -error)
    return -1;
  return 0;
}

} // namespace tf::exact
