/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./constants.hpp"
#include <cmath>

namespace tf {

template <typename T>
struct rad;

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

// Trigonometric functions accepting angle types
template <typename T>
auto sin(rad<T> angle) -> T { return std::sin(angle.value); }

template <typename T>
auto sin(deg<T> angle) -> T { return std::sin(rad<T>{angle}.value); }

template <typename T>
auto cos(rad<T> angle) -> T { return std::cos(angle.value); }

template <typename T>
auto cos(deg<T> angle) -> T { return std::cos(rad<T>{angle}.value); }

template <typename T>
auto tan(rad<T> angle) -> T { return std::tan(angle.value); }

template <typename T>
auto tan(deg<T> angle) -> T { return std::tan(rad<T>{angle}.value); }

} // namespace tf
