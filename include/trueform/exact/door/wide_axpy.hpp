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
#include "./round_to_wide.hpp"

#include <array>
#include <cstddef>

namespace tf::exact::door {

/// `a + m * b`, componentwise. The step is formed on the rung above the
/// triples and narrowed only when every component stands inside
/// @ref tf::exact::door::wide_placement_bound, so the multiplier may be
/// the whole clamp and the answer is never a wrapped one. False leaves
/// `out` untouched and the caller states the point by the rank below.
///
/// `out` may alias `a`: it is written only once the whole step is known.
template <typename Int>
auto wide_axpy(const std::array<typename tf::exact::meta<Int>::T1, 3> &a,
               typename tf::exact::meta<Int>::T1 m,
               const std::array<typename tf::exact::meta<Int>::T1, 3> &b,
               std::array<typename tf::exact::meta<Int>::T1, 3> &out) -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  const T2 bound = T2(wide_placement_bound<Int>());
  std::array<T2, 3> stepped{};
  for (std::size_t k = 0; k < 3; ++k) {
    stepped[k] = T2(a[k]) + T2(m) * T2(b[k]);
    if (stepped[k] > bound || stepped[k] < -bound)
      return false;
  }
  out = {static_cast<T1>(stepped[0]), static_cast<T1>(stepped[1]),
         static_cast<T1>(stepped[2])};
  return true;
}

} // namespace tf::exact::door
