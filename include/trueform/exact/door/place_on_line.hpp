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
#include "../meta.hpp"
#include "./ext_gcd.hpp"
#include "./place_near_line.hpp"
#include "./place_on_plane.hpp"
#include "./plane_frame.hpp"
#include "./quantized_plane.hpp"
#include "./round_to_wide.hpp"
#include "./wide_axpy.hpp"
#include "./wide_cross_magnitude.hpp"
#include "./wide_dot.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace tf::exact::door {

/// The rank-2 placement: the lattice point of two plane names nearest
/// `p`.
///
/// The rank-1 frame does the work twice: the vertex is put on the first
/// plane exactly, and the second plane is then a one-dimensional
/// Diophantine equation inside the first plane's own kernel lattice.
/// It is solvable exactly when the gcd of the second name against that
/// basis divides the residual; when it is, the common points form a
/// coset and the answer is its member nearest `p`. When it is not, no
/// lattice point lies on both planes at all and
/// @ref tf::exact::door::place_near_line states the pair's line
/// instead.
template <typename Int>
auto place_on_line(const quantized_plane<Int> &a, const quantized_plane<Int> &b,
                   const tf::point<Int, 3> &p,
                   std::array<typename tf::exact::meta<Int>::T1, 3> &out)
    -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  if (wide_cross_magnitude<Int>(a.normal, b.normal) == T2(0))
    return false;
  const T2 bound = T2(wide_placement_bound<Int>());
  const auto in_range = [bound](const T2 &v) {
    return v <= bound && v >= -bound;
  };

  const T2 residual_a = T2(a.offset) - wide_dot<Int>(a.normal, p);
  if (!in_range(residual_a))
    return place_near_line(a, b, p, out);

  const plane_frame<Int> frame = make_plane_frame<Int>(a.normal);
  std::array<T1, 3> step{};
  if (!place_on_plane<Int>(frame, static_cast<T1>(residual_a), step))
    return place_near_line(a, b, p, out);
  const std::array<T1, 3> on_a{T1(p[0]) + step[0], T1(p[1]) + step[1],
                               T1(p[2]) + step[2]};
  const T2 along1 = wide_dot<Int>(b.normal, frame.b1);
  const T2 along2 = wide_dot<Int>(b.normal, frame.b2);
  const T2 residual_b = T2(b.offset) - wide_dot<Int>(b.normal, on_a);
  if ((along1 == T2(0) && along2 == T2(0)) || !in_range(along1) ||
      !in_range(along2) || !in_range(residual_b))
    return place_near_line(a, b, p, out);

  T1 u(0), v(0);
  const T1 divisor =
      ext_gcd(static_cast<T1>(along1), static_cast<T1>(along2), u, v);
  if (divisor == T1(0) || static_cast<T1>(residual_b) % divisor != T1(0))
    return place_near_line(a, b, p, out);

  const T1 multiple = static_cast<T1>(residual_b) / divisor;
  const T2 toward1 = T2(multiple) * T2(u);
  const T2 toward2 = T2(multiple) * T2(v);
  std::array<T1, 3> base{};
  if (!in_range(toward1) || !in_range(toward2) ||
      !wide_axpy<Int>(on_a, static_cast<T1>(toward1), frame.b1, base) ||
      !wide_axpy<Int>(base, static_cast<T1>(toward2), frame.b2, base))
    return place_near_line(a, b, p, out);
  const T1 span1 = static_cast<T1>(along2) / divisor;
  const T1 span2 = static_cast<T1>(along1) / divisor;
  std::array<T1, 3> along{};
  for (std::size_t k = 0; k < 3; ++k) {
    const T2 component =
        T2(span1) * T2(frame.b1[k]) - T2(span2) * T2(frame.b2[k]);
    if (!in_range(component))
      return place_near_line(a, b, p, out);
    along[k] = static_cast<T1>(component);
  }
  const double length = static_cast<double>(wide_dot<Int>(along, along));
  if (!(length > 0.0))
    return place_near_line(a, b, p, out);

  const double toward =
      (static_cast<double>(T1(p[0])) - static_cast<double>(base[0])) *
          static_cast<double>(along[0]) +
      (static_cast<double>(T1(p[1])) - static_cast<double>(base[1])) *
          static_cast<double>(along[1]) +
      (static_cast<double>(T1(p[2])) - static_cast<double>(base[2])) *
          static_cast<double>(along[2]);
  const double at = toward / length;
  if (!std::isfinite(at))
    return place_near_line(a, b, p, out);
  const T1 nearest = round_to_wide<Int>(at);

  bool found = false;
  T2 best(0);
  std::array<T1, 3> candidate{};
  for (int offset = -2; offset <= 2; ++offset) {
    if (!wide_axpy<Int>(base, nearest + T1(offset), along, candidate))
      continue;
    const std::array<T1, 3> delta{candidate[0] - T1(p[0]),
                                  candidate[1] - T1(p[1]),
                                  candidate[2] - T1(p[2])};
    if (!in_range(T2(delta[0])) || !in_range(T2(delta[1])) ||
        !in_range(T2(delta[2])))
      continue;
    const T2 length2 = wide_dot<Int>(delta, delta);
    if (!found || length2 < best) {
      found = true;
      best = length2;
      out = candidate;
    }
  }
  return found ? true : place_near_line(a, b, p, out);
}

} // namespace tf::exact::door
