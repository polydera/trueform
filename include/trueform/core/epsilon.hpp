/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
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
template <typename T> inline constexpr T epsilon = core::epsilon<T>::make();

template <typename T> inline constexpr T epsilon2 = core::epsilon2<T>::make();
} // namespace tf
