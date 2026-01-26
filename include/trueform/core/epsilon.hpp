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
#include <type_traits>

namespace tf {
namespace core {
template <typename T> struct epsilon {
  static_assert(std::is_floating_point_v<T>, "Not supported");
};
template <> struct epsilon<float> {
  static constexpr auto make() -> float {
    return 0.00034526697709225118160247802734375;
  }
};
template <> struct epsilon<double> {
  static constexpr auto make() -> double { return 1.490116119384765625e-08; }
};

template <typename T> struct epsilon2 {
  static_assert(std::is_floating_point_v<T>, "Not supported");
};
template <> struct epsilon2<float> {
  static constexpr auto make() -> float { return 1.1920928955078125e-07; }
};
template <> struct epsilon2<double> {
  static constexpr auto make() -> double {
    return 2.220446049250313080847263336181640625e-16;
  }
};
} // namespace core

/// @ingroup core
/// @brief Geometric tolerance for floating-point comparisons.
///
/// Square root of machine epsilon, suitable for distance and
/// geometric comparisons where errors accumulate.
///
/// @tparam T The floating-point type (float or double).
template <typename T> inline constexpr T epsilon = core::epsilon<T>::make();

/// @ingroup core
/// @brief Machine epsilon for floating-point types.
///
/// Standard machine epsilon (1 ULP at 1.0). Use for precise
/// numerical comparisons.
///
/// @tparam T The floating-point type (float or double).
template <typename T> inline constexpr T epsilon2 = core::epsilon2<T>::make();

} // namespace tf
