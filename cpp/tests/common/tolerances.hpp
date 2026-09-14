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

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace tf::cpp::test {

template <typename Real> struct numeric_tolerance {
  Real absolute;
  Real relative;
};

template <typename Real>
constexpr auto base_tolerance() -> numeric_tolerance<Real> {
  static_assert(std::is_same<Real, float>::value ||
                    std::is_same<Real, double>::value,
                "test tolerances support float and double");
  if constexpr (std::is_same<Real, float>::value)
    return {Real{1e-5F}, Real{1e-5F}};
  else
    return {Real{1e-12}, Real{1e-12}};
}

template <typename Real>
auto scaled_tolerance(Real scale) -> numeric_tolerance<Real> {
  const auto base = base_tolerance<Real>();
  return {base.absolute * std::max(Real{1}, std::abs(scale)), base.relative};
}

template <typename Real>
auto within_tolerance(Real first, Real second, Real scale = Real{1}) -> bool {
  const auto tolerance = scaled_tolerance(scale);
  const auto magnitude = std::max(std::abs(first), std::abs(second));
  return std::abs(first - second) <=
         std::max(tolerance.absolute, tolerance.relative * magnitude);
}

} // namespace tf::cpp::test
