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

#include "../../core/point.hpp"
#include "../canonical_plane.hpp"
#include "./pool/exact_lane.hpp"
#include "./round_div.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace tf::exact::door {

/// The determinant of three exact normals, on the product rung.
///
/// A normal is a cross of coordinate differences, so it occupies twice a
/// difference; three of them, and an offset against two, stand inside the
/// product rung at both lattices — which is the rung's purpose.
template <typename Int>
auto exact_det3(
    const std::array<typename pool::exact_lane<Int>::coefficient_type, 3> &a,
    const std::array<typename pool::exact_lane<Int>::coefficient_type, 3> &b,
    const std::array<typename pool::exact_lane<Int>::coefficient_type, 3> &c) ->
    typename pool::exact_lane<Int>::product_type {
  using product_type = typename pool::exact_lane<Int>::product_type;
  const auto minor = [](const auto &x0, const auto &x1, const auto &y0,
                        const auto &y1) {
    return product_type(x0) * product_type(y1) -
           product_type(x1) * product_type(y0);
  };
  return product_type(a[0]) * minor(b[1], b[2], c[1], c[2]) -
         product_type(a[1]) * minor(b[0], b[2], c[0], c[2]) +
         product_type(a[2]) * minor(b[0], b[1], c[0], c[1]);
}

/// The lattice point nearest the exact rational meet of three planes —
/// Cramer on the product rung, then componentwise rounding. The answer is
/// ON the three planes exactly when the meet is itself a lattice point,
/// which the caller asks of it.
///
/// False when the planes are not independent or the meet leaves the
/// lattice.
template <typename Int>
auto place_at_exact_meet(const tf::exact::canonical_plane<Int> &a,
                         const tf::exact::canonical_plane<Int> &b,
                         const tf::exact::canonical_plane<Int> &c,
                         tf::point<Int, 3> &out) -> bool {
  using coefficient_type = typename pool::exact_lane<Int>::coefficient_type;
  using product_type = typename pool::exact_lane<Int>::product_type;

  const std::array<coefficient_type, 3> offsets{a[3], b[3], c[3]};
  const std::array<coefficient_type, 3> column0{a[0], b[0], c[0]};
  const std::array<coefficient_type, 3> column1{a[1], b[1], c[1]};
  const std::array<coefficient_type, 3> column2{a[2], b[2], c[2]};
  const product_type det = exact_det3<Int>(column0, column1, column2);
  if (det == product_type(0))
    return false;
  const product_type limit(std::numeric_limits<Int>::max());
  const product_type at[3] = {
      round_div(exact_det3<Int>(offsets, column1, column2), det),
      round_div(exact_det3<Int>(column0, offsets, column2), det),
      round_div(exact_det3<Int>(column0, column1, offsets), det)};
  for (std::size_t k = 0; k < 3; ++k)
    if (at[k] > limit || at[k] < -limit)
      return false;
  out = tf::point<Int, 3>{static_cast<Int>(at[0]), static_cast<Int>(at[1]),
                          static_cast<Int>(at[2])};
  return true;
}

} // namespace tf::exact::door
