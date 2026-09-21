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

namespace tf::cpp {

template <typename T>
auto mat_mul(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T>;
template <typename T> auto inverted(const nd_array<T> &matrix) -> nd_array<T>;
template <typename T>
auto dot(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T>;
template <typename T>
auto cross(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T>;
template <typename T> auto normalize(const nd_array<T> &a) -> nd_array<T>;
template <typename T>
auto normalize(const nd_array<T> &a, int axis) -> nd_array<T>;
template <typename T> auto normalize_inplace(nd_array<T> &a) -> void;
template <typename T> auto normalize_inplace(nd_array<T> &a, int axis) -> void;

#define TF_CPP_EXTERN_VECTOR(T)                                                \
  extern template auto mat_mul<T>(const nd_array<T> &, const nd_array<T> &)    \
      -> nd_array<T>;                                                          \
  extern template auto dot<T>(const nd_array<T> &, const nd_array<T> &)        \
      -> nd_array<T>;                                                          \
  extern template auto cross<T>(const nd_array<T> &, const nd_array<T> &)      \
      -> nd_array<T>

#define TF_CPP_EXTERN_NORMALIZE(T)                                             \
  extern template auto normalize<T>(const nd_array<T> &) -> nd_array<T>;       \
  extern template auto normalize<T>(const nd_array<T> &, int) -> nd_array<T>;  \
  extern template auto normalize_inplace<T>(nd_array<T> &) -> void;            \
  extern template auto normalize_inplace<T>(nd_array<T> &, int) -> void

TF_CPP_EXTERN_VECTOR(std::int32_t);
TF_CPP_EXTERN_VECTOR(std::int64_t);
TF_CPP_EXTERN_VECTOR(float);
TF_CPP_EXTERN_VECTOR(double);
TF_CPP_EXTERN_NORMALIZE(float);
TF_CPP_EXTERN_NORMALIZE(double);

extern template auto inverted<float>(const nd_array<float> &)
    -> nd_array<float>;
extern template auto inverted<double>(const nd_array<double> &)
    -> nd_array<double>;

#undef TF_CPP_EXTERN_NORMALIZE
#undef TF_CPP_EXTERN_VECTOR

} // namespace tf::cpp
