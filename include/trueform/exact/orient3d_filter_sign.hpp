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
#include "./vertex.hpp"

namespace tf::exact {

/// The sign of `orient3d_value_scaled(a, b, c, d)` decided from the double
/// images of the coordinate differences, `0` where the filter cannot decide.
/// The images are exact below 2^52 and rounded to nearest above it; the bound
/// is (7 + 56e)e times the permanent for exact images, grown by one rounding
/// per factor of each product for rounded ones.
template <typename Coord>
auto orient3d_filter_sign(const pt3<Coord> &a, const pt3<Coord> &b,
                          const pt3<Coord> &c, const pt3<Coord> &d) -> int {
  double ax, ay, az, bx, by, bz, cx, cy, cz;
  const difference_image_kind kinds[9] = {
      difference_image(b[0], a[0], ax), difference_image(b[1], a[1], ay),
      difference_image(b[2], a[2], az), difference_image(c[0], a[0], bx),
      difference_image(c[1], a[1], by), difference_image(c[2], a[2], bz),
      difference_image(d[0], a[0], cx), difference_image(d[1], a[1], cy),
      difference_image(d[2], a[2], cz)};
  bool rounded = false;
  for (const auto kind : kinds) {
    if (kind == difference_image_kind::refused)
      return 0;
    rounded = rounded || kind == difference_image_kind::rounded;
  }
  const double bycz = by * cz, bzcy = bz * cy;
  const double bxcz = bx * cz, bzcx = bz * cx;
  const double bxcy = bx * cy, bycx = by * cx;
  const double determinant =
      ax * (bycz - bzcy) - ay * (bxcz - bzcx) + az * (bxcy - bycx);
  const auto magnitude = [](double v) { return v < 0 ? -v : v; };
  const double permanent = (magnitude(bycz) + magnitude(bzcy)) * magnitude(ax) +
                           (magnitude(bxcz) + magnitude(bzcx)) * magnitude(ay) +
                           (magnitude(bxcy) + magnitude(bycx)) * magnitude(az);
  constexpr double epsilon = 1.1102230246251565e-16;
  constexpr double exact_coefficient = (7.0 + 56.0 * epsilon) * epsilon;
  constexpr double rounded_coefficient = (10.0 + 256.0 * epsilon) * epsilon;
  const double error =
      (rounded ? rounded_coefficient : exact_coefficient) * permanent;
  if (determinant > error)
    return 1;
  if (determinant < -error)
    return -1;
  return 0;
}

} // namespace tf::exact
