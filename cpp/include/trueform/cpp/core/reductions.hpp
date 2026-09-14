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

#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>
#include <type_traits>

namespace tf::cpp {

template <typename T>
auto sum(const nd_array<T> &array)
    -> std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T>;

template <typename T>
auto sum(const nd_array<T> &array, int axis) -> nd_array<
    std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T>>;

template <typename T> auto min(const nd_array<T> &array) -> T;
template <typename T>
auto min(const nd_array<T> &array, int axis) -> nd_array<T>;

template <typename T> auto max(const nd_array<T> &array) -> T;
template <typename T>
auto max(const nd_array<T> &array, int axis) -> nd_array<T>;

template <typename T> auto mean(const nd_array<T> &array) -> double;
template <typename T>
auto mean(const nd_array<T> &array, int axis)
    -> nd_array<std::conditional_t<std::is_same_v<T, double>, double, float>>;

template <typename T>
auto norm(const nd_array<T> &array)
    -> std::conditional_t<std::is_same_v<T, double>, double, float>;
template <typename T>
auto norm(const nd_array<T> &array, int axis)
    -> nd_array<std::conditional_t<std::is_same_v<T, double>, double, float>>;

template <typename T> auto argmin(const nd_array<T> &array) -> std::int32_t;
template <typename T>
auto argmin(const nd_array<T> &array, int axis) -> nd_array<std::int32_t>;

template <typename T> auto argmax(const nd_array<T> &array) -> std::int32_t;
template <typename T>
auto argmax(const nd_array<T> &array, int axis) -> nd_array<std::int32_t>;

auto any(const nd_array<std::int8_t> &array) -> int;
auto any(const nd_array<std::int8_t> &array, int axis) -> nd_array<std::int8_t>;

auto all(const nd_array<std::int8_t> &array) -> int;
auto all(const nd_array<std::int8_t> &array, int axis) -> nd_array<std::int8_t>;

#define TF_CPP_EXTERN_REDUCTIONS(T)                                            \
  extern template auto sum<T>(const nd_array<T> &)                             \
      -> std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T>;  \
  extern template auto sum<T>(const nd_array<T> &, int) -> nd_array<           \
      std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T>>;    \
  extern template auto min<T>(const nd_array<T> &) -> T;                       \
  extern template auto min<T>(const nd_array<T> &, int) -> nd_array<T>;        \
  extern template auto max<T>(const nd_array<T> &) -> T;                       \
  extern template auto max<T>(const nd_array<T> &, int) -> nd_array<T>;        \
  extern template auto mean<T>(const nd_array<T> &) -> double;                 \
  extern template auto mean<T>(const nd_array<T> &, int) -> nd_array<          \
      std::conditional_t<std::is_same_v<T, double>, double, float>>;           \
  extern template auto norm<T>(const nd_array<T> &)                            \
      -> std::conditional_t<std::is_same_v<T, double>, double, float>;         \
  extern template auto norm<T>(const nd_array<T> &, int) -> nd_array<          \
      std::conditional_t<std::is_same_v<T, double>, double, float>>;           \
  extern template auto argmin<T>(const nd_array<T> &) -> std::int32_t;         \
  extern template auto argmin<T>(const nd_array<T> &, int)                     \
      -> nd_array<std::int32_t>;                                               \
  extern template auto argmax<T>(const nd_array<T> &) -> std::int32_t;         \
  extern template auto argmax<T>(const nd_array<T> &, int)                     \
      -> nd_array<std::int32_t>

TF_CPP_EXTERN_REDUCTIONS(std::int8_t);
TF_CPP_EXTERN_REDUCTIONS(std::int32_t);
TF_CPP_EXTERN_REDUCTIONS(float);
TF_CPP_EXTERN_REDUCTIONS(double);

#undef TF_CPP_EXTERN_REDUCTIONS

} // namespace tf::cpp
