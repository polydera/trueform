/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
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

template <typename T> inline constexpr T pi = core::pi_v<T>::make();
template <typename T> inline constexpr T two_pi = core::two_pi_v<T>::make();

} // namespace tf
