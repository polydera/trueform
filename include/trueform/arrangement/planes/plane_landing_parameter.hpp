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

#include "../../exact/meta.hpp"
#include "../../exact/wide_dyadic_ratio.hpp"
#include <cstddef>

namespace tf::arrangement {

/// Where a point lands on the segment `a b` of a plane's projection: the
/// dyadic position of its foot, or `-1` where the foot does not fall
/// strictly inside the span or its position rounds to an end. A landing at
/// an end is a vertex fact, not an edge landing, so it is never stated as
/// one.
template <typename Int, typename A, typename B, typename P>
auto plane_landing_parameter(const A &a, const B &b, const P &p) ->
    typename tf::exact::meta<Int>::param_type {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const T1 dx = T1(b[0]) - T1(a[0]), dy = T1(b[1]) - T1(a[1]);
  const T1 qx = T1(p[0]) - T1(a[0]), qy = T1(p[1]) - T1(a[1]);
  const T2 along = T2(qx) * T2(dx) + T2(qy) * T2(dy);
  const T2 length = T2(dx) * T2(dx) + T2(dy) * T2(dy);
  if (!(along > T2(0)) || !(along < length))
    return param_t(-1);
  const auto parameter = tf::exact::wide_dyadic_ratio<Int>(along, length);
  const param_t whole = param_t(1) << tf::exact::meta<Int>::param_bits;
  if (parameter <= param_t(0) || parameter >= whole)
    return param_t(-1);
  return parameter;
}

/// Where a point standing within one lattice unit of the line `a b` lands on
/// the segment: @ref plane_landing_parameter where the point's distance to the
/// line is below one unit, `-1` otherwise. On the lattice such a point IS on
/// the segment, since no point between it and the line can be stated.
template <typename Int, typename A, typename B, typename P>
auto plane_unit_landing_parameter(const A &a, const B &b, const P &p) ->
    typename tf::exact::meta<Int>::param_type {
  using T2 = typename tf::exact::meta<Int>::T2;
  using T3 = typename tf::exact::meta<Int>::T3;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const T2 dx = T2(b[0]) - T2(a[0]);
  const T2 dy = T2(b[1]) - T2(a[1]);
  const T2 length2 = dx * dx + dy * dy;
  const T3 off = T3(dx * (T2(p[1]) - T2(a[1])) - dy * (T2(p[0]) - T2(a[0])));
  if (off * off >= T3(length2))
    return param_t(-1);
  return plane_landing_parameter<Int>(a, b, p);
}

/// WHERE A LANDING STANDS ALONG ITS ROW IN SPACE. A plane's projection
/// can hold a point on a side that another plane's projection orders the
/// other way round, when the point and the side's end stand closer along
/// the row than the lattice can tell apart; each plane would then land the
/// point before the other's end, round after round. The row is one line in
/// space, so the order along it is read there, once, off the lattice
/// positions: a point within the rounding radius of an end along the row
/// IS that end, a vertex fact the landing states as a weld; every other
/// point the projection put on the row is a landing, CLIPPED ONTO THE ROW
/// for the plane's own work by the parameter the plane saw, its own
/// position untouched. A plane that saw a junction states it; none is
/// dropped.
enum class plane_row_landing { at_start, at_end, inside };

/// Two lattice points stand within sqrt(3) of one exact position, so an
/// order along a row is a fact only past that: the radius squared.
inline constexpr int plane_row_rounding_radius2 = 3;

template <typename Int, typename A, typename B, typename P>
auto plane_row_landing_of(const A &a, const B &b, const P &p)
    -> plane_row_landing {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using T3 = typename tf::exact::meta<Int>::T3;
  T2 along(0), length2(0);
  for (std::size_t k = 0; k < 3; ++k) {
    const T1 d = T1(b[k]) - T1(a[k]);
    const T1 q = T1(p[k]) - T1(a[k]);
    along = along + T2(d) * T2(q);
    length2 = length2 + T2(d) * T2(d);
  }
  const T3 radius2(T3(plane_row_rounding_radius2) * T3(length2));
  if (T3(along) * T3(along) < radius2)
    return plane_row_landing::at_start;
  const T2 rest = length2 - along;
  if (T3(rest) * T3(rest) < radius2)
    return plane_row_landing::at_end;
  return plane_row_landing::inside;
}

} // namespace tf::arrangement
