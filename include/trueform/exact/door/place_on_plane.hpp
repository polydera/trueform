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
#include "./plane_frame.hpp"
#include "./round_to_wide.hpp"
#include "./wide_axpy.hpp"
#include "./wide_dot.hpp"

#include <array>
#include <cstddef>

namespace tf::exact::door {

/// The rank-1 placement: the displacement from `p` to a nearest lattice
/// point of the plane `N . x = N . p + r`, stated in the frame of that
/// direction. The Bezout vector reaches the plane, the reduced kernel
/// basis walks it, and the residual is driven to zero exactly, so the
/// answer lies on its plane and not near it. The Gram solve and the
/// sweep after it only choose which lattice point; the plane membership
/// is integral throughout.
///
/// The frame is the caller's, because the rank above solves inside the
/// same one and a Lagrange reduction is not run twice on one direction.
///
/// False when the step onto the plane leaves the rung the solve is
/// stated on: the residual grows with the square of the band and the
/// Bezout vector with the band, so a wide enough band asks for a product
/// no rung holds. It refuses rather than wraps, and the caller states
/// the vertex by the rank below — @ref tf::exact::door::place_on_line
/// by its own line, @ref tf::exact::door::place_vertex by leaving the
/// vertex where it was. The shortening sweep refuses only the candidate
/// it cannot form; the answer it already holds stands.
template <typename Int>
auto place_on_plane(const plane_frame<Int> &frame,
                    typename tf::exact::meta<Int>::T1 r,
                    std::array<typename tf::exact::meta<Int>::T1, 3> &out)
    -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  if (r == T1(0)) {
    out = {T1(0), T1(0), T1(0)};
    return true;
  }
  const auto &b1 = frame.b1;
  const auto &b2 = frame.b2;
  const T2 bound = T2(wide_placement_bound<Int>());
  const std::array<T2, 3> onto{T2(r) * T2(frame.s[0]), T2(r) * T2(frame.s[1]),
                               T2(r) * T2(frame.s[2])};
  for (std::size_t k = 0; k < 3; ++k)
    if (onto[k] > bound || onto[k] < -bound)
      return false;
  std::array<T1, 3> at{static_cast<T1>(onto[0]), static_cast<T1>(onto[1]),
                       static_cast<T1>(onto[2])};
  {
    const double g00 = static_cast<double>(wide_dot<Int>(b1, b1));
    const double g11 = static_cast<double>(wide_dot<Int>(b2, b2));
    const double g01 = static_cast<double>(wide_dot<Int>(b1, b2));
    const double det = g00 * g11 - g01 * g01;
    if (det > 0.0) {
      const double c0 = static_cast<double>(wide_dot<Int>(at, b1));
      const double c1 = static_cast<double>(wide_dot<Int>(at, b2));
      if (!wide_axpy<Int>(at, round_to_wide<Int>((-c0 * g11 + c1 * g01) / det),
                          b1, at) ||
          !wide_axpy<Int>(at, round_to_wide<Int>((-c1 * g00 + c0 * g01) / det),
                          b2, at))
        return false;
    }
  }
  for (int step = 0; step < 4; ++step) {
    const T2 n1 = wide_dot<Int>(b1, b1);
    const T2 n2 = wide_dot<Int>(b2, b2);
    if (n1 > T2(0) &&
        !wide_axpy<Int>(
            at,
            round_to_wide<Int>(-static_cast<double>(wide_dot<Int>(at, b1)) /
                               static_cast<double>(n1)),
            b1, at))
      return false;
    if (n2 > T2(0) &&
        !wide_axpy<Int>(
            at,
            round_to_wide<Int>(-static_cast<double>(wide_dot<Int>(at, b2)) /
                               static_cast<double>(n2)),
            b2, at))
      return false;
  }
  out = at;
  T2 best_length = wide_dot<Int>(at, at);
  std::array<T1, 3> shifted{}, candidate{};
  for (int x = -3; x <= 3; ++x)
    for (int y = -3; y <= 3; ++y) {
      if (!wide_axpy<Int>(at, T1(x), b1, shifted) ||
          !wide_axpy<Int>(shifted, T1(y), b2, candidate))
        continue;
      const T2 length = wide_dot<Int>(candidate, candidate);
      if (length < best_length) {
        best_length = length;
        out = candidate;
      }
    }
  return true;
}

} // namespace tf::exact::door
