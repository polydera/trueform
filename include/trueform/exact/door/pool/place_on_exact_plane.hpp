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

#include "./exact_axpy.hpp"
#include "./exact_dot.hpp"
#include "./exact_lane.hpp"
#include "./exact_placement_bound.hpp"
#include "./exact_plane_frame.hpp"
#include "./reduce_on_product_rung.hpp"
#include "./round_to_exact.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace tf::exact::door::pool {

/// The rank-1 placement against an EXACT plane: the displacement from `p`
/// to a nearest lattice point of `N . x = N . p + r`, stated in the frame
/// of that direction. The Bezout vector reaches the plane, the reduced
/// kernel basis walks it, and the residual is driven to zero exactly, so
/// the answer lies ON its plane; the Gram solve and the sweep after it
/// only choose which lattice point.
///
/// THE LIFT IS FORMED ON THE PRODUCT RUNG. `r * s` is the unreduced reach
/// onto the plane and stands a rung above the coefficient one; the kernel
/// basis walks it back down. Only the walked value is asked to stand on the
/// coefficient rung, which is where the bound is a real invariant.
///
/// False when the reduced lift still leaves the coefficient rung. It is the
/// bounded search's own refusal and states nothing about whether a
/// one-tolerance lattice point exists.
template <typename Int>
auto place_on_exact_plane(
    const exact_plane_frame<Int> &frame,
    typename exact_lane<Int>::coefficient_type r,
    std::array<typename exact_lane<Int>::coefficient_type, 3> &out) -> bool {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using product_type = typename exact_lane<Int>::product_type;

  if (r == coefficient_type(0)) {
    out = {coefficient_type(0), coefficient_type(0), coefficient_type(0)};
    return true;
  }
  if (!frame.available)
    return false;

  const auto &b1 = frame.b1;
  const auto &b2 = frame.b2;
  const product_type bound(exact_placement_bound<Int>());
  std::array<product_type, 3> lifted{};
  for (std::size_t k = 0; k < 3; ++k)
    lifted[k] = product_type(r) * product_type(frame.s[k]);
  const auto stands = [&bound](const std::array<product_type, 3> &v) {
    return !(v[0] > bound || v[0] < -bound || v[1] > bound || v[1] < -bound ||
             v[2] > bound || v[2] < -bound);
  };
  for (int pass = 0; pass < 3 && !stands(lifted); ++pass) {
    reduce_on_product_rung<Int>(lifted, b1);
    reduce_on_product_rung<Int>(lifted, b2);
  }
  if (!stands(lifted))
    return false;
  std::array<coefficient_type, 3> at{
      static_cast<coefficient_type>(lifted[0]),
      static_cast<coefficient_type>(lifted[1]),
      static_cast<coefficient_type>(lifted[2])};
  {
    const double g00 = static_cast<double>(exact_dot<Int>(b1, b1));
    const double g11 = static_cast<double>(exact_dot<Int>(b2, b2));
    const double g01 = static_cast<double>(exact_dot<Int>(b1, b2));
    const double det = g00 * g11 - g01 * g01;
    if (det > 0.0) {
      const double c0 = static_cast<double>(exact_dot<Int>(at, b1));
      const double c1 = static_cast<double>(exact_dot<Int>(at, b2));
      const double along_b1 = (-c0 * g11 + c1 * g01) / det;
      const double along_b2 = (-c1 * g00 + c0 * g01) / det;
      if (!std::isfinite(along_b1) || !std::isfinite(along_b2))
        return false;
      if (!exact_axpy<Int>(at, round_to_exact<Int>(along_b1), b1, at) ||
          !exact_axpy<Int>(at, round_to_exact<Int>(along_b2), b2, at))
        return false;
    }
  }
  for (int step = 0; step < 4; ++step) {
    const product_type n1 = exact_dot<Int>(b1, b1);
    const product_type n2 = exact_dot<Int>(b2, b2);
    if (n1 > product_type(0)) {
      const double along = -static_cast<double>(exact_dot<Int>(at, b1)) /
                           static_cast<double>(n1);
      if (!std::isfinite(along) ||
          !exact_axpy<Int>(at, round_to_exact<Int>(along), b1, at))
        return false;
    }
    if (n2 > product_type(0)) {
      const double along = -static_cast<double>(exact_dot<Int>(at, b2)) /
                           static_cast<double>(n2);
      if (!std::isfinite(along) ||
          !exact_axpy<Int>(at, round_to_exact<Int>(along), b2, at))
        return false;
    }
  }
  out = at;
  product_type best_length = exact_dot<Int>(at, at);
  std::array<coefficient_type, 3> shifted{}, candidate{};
  for (int x = -3; x <= 3; ++x)
    for (int y = -3; y <= 3; ++y) {
      if (!exact_axpy<Int>(at, coefficient_type(x), b1, shifted) ||
          !exact_axpy<Int>(shifted, coefficient_type(y), b2, candidate))
        continue;
      const product_type length = exact_dot<Int>(candidate, candidate);
      if (length < best_length) {
        best_length = length;
        out = candidate;
      }
    }
  return true;
}

} // namespace tf::exact::door::pool
