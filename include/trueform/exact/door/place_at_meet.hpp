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
#include "./quantized_plane.hpp"
#include "./round_div.hpp"
#include "./round_to_wide.hpp"
#include "./wide_det3.hpp"

#include <array>

namespace tf::exact::door {

/// The rank-3 placement: the lattice point nearest the exact rational
/// meet of three plane names — Cramer on the rung above the names, then
/// componentwise rounding, which is the nearest point of the integer
/// lattice to any real point.
///
/// Nothing but the three names enters, so two vertices of two different
/// forms whose incident faces carry the same three names land on the
/// same integer and the forms meet by identity rather than by
/// proximity.
///
/// False when the three names are not independent, or when the meet
/// leaves the rung the placement is solved on.
template <typename Int>
auto place_at_meet(const quantized_plane<Int> &a, const quantized_plane<Int> &b,
                   const quantized_plane<Int> &c,
                   std::array<typename tf::exact::meta<Int>::T1, 3> &out)
    -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  const std::array<T1, 3> offsets{a.offset, b.offset, c.offset};
  const std::array<T1, 3> column0{a.normal[0], b.normal[0], c.normal[0]};
  const std::array<T1, 3> column1{a.normal[1], b.normal[1], c.normal[1]};
  const std::array<T1, 3> column2{a.normal[2], b.normal[2], c.normal[2]};

  const T2 det = wide_det3<Int>(column0, column1, column2);
  if (det == T2(0))
    return false;
  const T2 x = round_div(wide_det3<Int>(offsets, column1, column2), det);
  const T2 y = round_div(wide_det3<Int>(column0, offsets, column2), det);
  const T2 z = round_div(wide_det3<Int>(column0, column1, offsets), det);
  const T2 bound = T2(wide_placement_bound<Int>());
  if (x > bound || x < -bound || y > bound || y < -bound || z > bound ||
      z < -bound)
    return false;
  out = {static_cast<T1>(x), static_cast<T1>(y), static_cast<T1>(z)};
  return true;
}

} // namespace tf::exact::door
