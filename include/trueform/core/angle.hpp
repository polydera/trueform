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
#include "./constants.hpp"
#include <cmath>

namespace tf {

template <typename T>
struct rad;

/// @ingroup core_primitives
/// @brief Angle in degrees with type-safe arithmetic.
///
/// Provides a strongly-typed angle representation in degrees.
/// Implicitly converts to @ref tf::rad. Supports arithmetic
/// operations and comparisons.
///
/// @tparam T The underlying numeric type.
template <typename T>
struct deg {
  using value_type = T;
  T value;

  constexpr deg() = default;
  constexpr explicit deg(T v) : value{v} {}

  // Convert to radians
  constexpr operator rad<T>() const;

  // Arithmetic
  friend constexpr deg operator-(deg a) { return deg{-a.value}; }
  friend constexpr deg operator+(deg a, deg b) { return deg{a.value + b.value}; }
  friend constexpr deg operator-(deg a, deg b) { return deg{a.value - b.value}; }
  friend constexpr deg operator*(deg a, T s) { return deg{a.value * s}; }
  friend constexpr deg operator*(T s, deg a) { return deg{s * a.value}; }
  friend constexpr deg operator/(deg a, T s) { return deg{a.value / s}; }

  // Comparisons
  friend constexpr bool operator==(deg a, deg b) { return a.value == b.value; }
  friend constexpr bool operator!=(deg a, deg b) { return a.value != b.value; }
  friend constexpr bool operator<(deg a, deg b) { return a.value < b.value; }
  friend constexpr bool operator<=(deg a, deg b) { return a.value <= b.value; }
  friend constexpr bool operator>(deg a, deg b) { return a.value > b.value; }
  friend constexpr bool operator>=(deg a, deg b) { return a.value >= b.value; }
};

/// @ingroup core_primitives
/// @brief Angle in radians with type-safe arithmetic.
///
/// Provides a strongly-typed angle representation in radians.
/// Implicitly converts to @ref tf::deg. Supports arithmetic
/// operations and comparisons.
///
/// @tparam T The underlying numeric type.
template <typename T>
struct rad {
  using value_type = T;
  T value;

  constexpr rad() = default;
  constexpr explicit rad(T v) : value{v} {}

  // Convert to degrees
  constexpr operator deg<T>() const;

  // Arithmetic
  friend constexpr rad operator-(rad a) { return rad{-a.value}; }
  friend constexpr rad operator+(rad a, rad b) { return rad{a.value + b.value}; }
  friend constexpr rad operator-(rad a, rad b) { return rad{a.value - b.value}; }
  friend constexpr rad operator*(rad a, T s) { return rad{a.value * s}; }
  friend constexpr rad operator*(T s, rad a) { return rad{s * a.value}; }
  friend constexpr rad operator/(rad a, T s) { return rad{a.value / s}; }

  // Comparisons
  friend constexpr bool operator==(rad a, rad b) { return a.value == b.value; }
  friend constexpr bool operator!=(rad a, rad b) { return a.value != b.value; }
  friend constexpr bool operator<(rad a, rad b) { return a.value < b.value; }
  friend constexpr bool operator<=(rad a, rad b) { return a.value <= b.value; }
  friend constexpr bool operator>(rad a, rad b) { return a.value > b.value; }
  friend constexpr bool operator>=(rad a, rad b) { return a.value >= b.value; }
};

// Conversion implementations
template <typename T>
constexpr deg<T>::operator rad<T>() const {
  return rad<T>{value * pi<T> / T{180}};
}

template <typename T>
constexpr rad<T>::operator deg<T>() const {
  return deg<T>{value * T{180} / pi<T>};
}

/// @ingroup core_primitives
/// @brief Compute sine of an angle in radians.
/// @tparam T The numeric type.
/// @param angle The angle in radians.
/// @return The sine value.
template <typename T>
auto sin(rad<T> angle) -> T { return std::sin(angle.value); }

/// @ingroup core_primitives
/// @brief Compute sine of an angle in degrees.
/// @overload
template <typename T>
auto sin(deg<T> angle) -> T { return std::sin(rad<T>{angle}.value); }

/// @ingroup core_primitives
/// @brief Compute cosine of an angle in radians.
/// @tparam T The numeric type.
/// @param angle The angle in radians.
/// @return The cosine value.
template <typename T>
auto cos(rad<T> angle) -> T { return std::cos(angle.value); }

/// @ingroup core_primitives
/// @brief Compute cosine of an angle in degrees.
/// @overload
template <typename T>
auto cos(deg<T> angle) -> T { return std::cos(rad<T>{angle}.value); }

/// @ingroup core_primitives
/// @brief Compute tangent of an angle in radians.
/// @tparam T The numeric type.
/// @param angle The angle in radians.
/// @return The tangent value.
template <typename T>
auto tan(rad<T> angle) -> T { return std::tan(angle.value); }

/// @ingroup core_primitives
/// @brief Compute tangent of an angle in degrees.
/// @overload
template <typename T>
auto tan(deg<T> angle) -> T { return std::tan(rad<T>{angle}.value); }

} // namespace tf
