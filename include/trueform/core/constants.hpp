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

template <typename T> struct pi_v {
  static_assert(std::is_floating_point_v<T>, "Not supported");
};
template <> struct pi_v<float> {
  static constexpr auto make() -> float { return 3.14159265358979323846f; }
};
template <> struct pi_v<double> {
  static constexpr auto make() -> double { return 3.14159265358979323846; }
};

template <typename T> struct two_pi_v {
  static_assert(std::is_floating_point_v<T>, "Not supported");
};
template <> struct two_pi_v<float> {
  static constexpr auto make() -> float { return 6.28318530717958647692f; }
};
template <> struct two_pi_v<double> {
  static constexpr auto make() -> double { return 6.28318530717958647692; }
};

} // namespace core

/// @ingroup core
/// @brief Pi constant for floating-point types.
///
/// Provides the mathematical constant pi with full precision
/// for float and double types.
///
/// @tparam T The floating-point type (float or double).
template <typename T> inline constexpr T pi = core::pi_v<T>::make();

/// @ingroup core
/// @brief Two-pi constant for floating-point types.
///
/// Provides 2*pi with full precision for float and double types.
/// Useful for angle wrapping and full-circle calculations.
///
/// @tparam T The floating-point type (float or double).
template <typename T> inline constexpr T two_pi = core::two_pi_v<T>::make();

} // namespace tf
