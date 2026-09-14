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

#include "./exact_lane.hpp"
#include "./exact_placement_bound.hpp"

#include <array>
#include <cstddef>

namespace tf::exact::door::pool {

/// `a + m * b`, componentwise, formed on the product rung and narrowed
/// only when every component stands inside
/// @ref tf::exact::door::pool::exact_placement_bound. False leaves `out`
/// untouched; `out` may alias `a`.
template <typename Int>
auto exact_axpy(
    const std::array<typename exact_lane<Int>::coefficient_type, 3> &a,
    typename exact_lane<Int>::coefficient_type m,
    const std::array<typename exact_lane<Int>::coefficient_type, 3> &b,
    std::array<typename exact_lane<Int>::coefficient_type, 3> &out) -> bool {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using product_type = typename exact_lane<Int>::product_type;

  const product_type bound(exact_placement_bound<Int>());
  std::array<product_type, 3> stepped{};
  for (std::size_t k = 0; k < 3; ++k) {
    stepped[k] = product_type(a[k]) + product_type(m) * product_type(b[k]);
    if (stepped[k] > bound || stepped[k] < -bound)
      return false;
  }
  out = {static_cast<coefficient_type>(stepped[0]),
         static_cast<coefficient_type>(stepped[1]),
         static_cast<coefficient_type>(stepped[2])};
  return true;
}

} // namespace tf::exact::door::pool
