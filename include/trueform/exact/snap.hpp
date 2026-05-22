/*
 * Copyright (c) 2025 XLAB
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

#include "../core/coordinate_type.hpp"
#include "../core/point.hpp"
#include "../core/point_like.hpp"
#include <cmath>
#include <type_traits>

namespace tf::exact {

/// @ingroup exact_snap
/// @brief Round-to-nearest snap key (bucket index) on a grid.
///        Integer `T`: cell width is `tol + 1`, pure integer arithmetic.
///        Floating-point `T`: cell width is `tol`, key = `llround(x / tol)`.
template <typename T>
constexpr auto snap_key(T x, T tol) {
  if constexpr (std::is_integral_v<T>) {
    static_assert(std::is_signed_v<T>,
                  "snap_key requires a signed integer type for integer input");
    T divisor = tol + T(1);
    T q = x / divisor;
    T r = x - q * divisor;
    T half = divisor / T(2);
    if (r >= half)
      ++q;
    else if (r <= -half)
      --q;
    return q;
  } else {
    return std::llround(static_cast<double>(x) / static_cast<double>(tol));
  }
}

/// @ingroup exact_snap
/// @brief Snapped coordinate.
///        Integer: `snap_key(x, tol) * (tol + 1)`. Float: `snap_key(x, tol) * tol`.
template <typename T>
constexpr auto snap_coord(T x, T tol) -> T {
  if constexpr (std::is_integral_v<T>)
    return snap_key(x, tol) * (tol + T(1));
  else
    return static_cast<T>(snap_key(x, tol)) * tol;
}

/// @ingroup exact_snap
/// @brief Per-coordinate snap of a point.
template <std::size_t Dims, typename Policy>
constexpr auto snap_point(const tf::point_like<Dims, Policy> &p,
                          tf::coordinate_type<Policy> tol) {
  using T = tf::coordinate_type<Policy>;
  tf::point<T, Dims> q;
  for (std::size_t d = 0; d < Dims; ++d)
    q[d] = snap_coord<T>(p[d], tol);
  return q;
}

} // namespace tf::exact
