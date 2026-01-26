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
#include <cmath>
namespace tf {
/// @ingroup core_algorithms
/// @brief Computes a safe inverse of a floating-point value, guarding against division by values near zero.
///
/// If the absolute value of the input `t` is smaller than the specified `epsilon`, the function
/// returns positive or negative infinity, depending on the sign of `t`. Otherwise, it returns the
/// standard reciprocal `1 / t`.
///
/// This is useful in geometric algorithms (e.g., ray casting) where small or zero values
/// can lead to unstable results or undefined behavior during inversion.
///
/// @tparam T A floating-point type (e.g., float, double).
/// @param t The value to invert.
/// @param epsilon The threshold below which `t` is considered too small for safe inversion.
/// @return `1 / t` if `|t| >= epsilon`; otherwise, `+∞` or `−∞` depending on the sign of `t`.
template <typename T> auto epsilon_inverse(const T &t, T epsilon) {
  if (std::abs(t) < epsilon) {
    return std::numeric_limits<T>::infinity() *
           std::copysign(static_cast<T>(1), t);
  } else {
    return static_cast<T>(1) / t;
  }
}
/// @ingroup core_algorithms
/// @brief Computes a safe inverse of a floating-point value using the machine epsilon for the type.
///
/// Equivalent to calling `epsilon_inverse(t, std::numeric_limits<T>::epsilon())`.
///
/// @tparam T A floating-point type (e.g., float, double).
/// @param t The value to invert.
/// @return `1 / t` if `|t| >= ε`; otherwise, `+∞` or `−∞` depending on the sign of `t`.
template <typename T> auto epsilon_inverse(const T &t) {
  return epsilon_inverse(t, std::numeric_limits<T>::epsilon());
}
} // namespace tf
