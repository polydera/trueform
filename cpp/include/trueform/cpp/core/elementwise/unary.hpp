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

#include <cmath>
#include <cstdint>

namespace tf::cpp {

#define TF_CPP_DECLARE_UNARY(NAME)                                             \
  template <typename T> auto NAME(const nd_array<T> &a) -> nd_array<T>;        \
  template <typename T> auto NAME##_inplace(nd_array<T> &a) -> void

TF_CPP_DECLARE_UNARY(sqrt);
TF_CPP_DECLARE_UNARY(sin);
TF_CPP_DECLARE_UNARY(cos);
TF_CPP_DECLARE_UNARY(tan);
TF_CPP_DECLARE_UNARY(asin);
TF_CPP_DECLARE_UNARY(acos);
TF_CPP_DECLARE_UNARY(atan);
TF_CPP_DECLARE_UNARY(exp);
TF_CPP_DECLARE_UNARY(log);
TF_CPP_DECLARE_UNARY(log2);
TF_CPP_DECLARE_UNARY(log10);
TF_CPP_DECLARE_UNARY(floor);
TF_CPP_DECLARE_UNARY(ceil);
TF_CPP_DECLARE_UNARY(round);
TF_CPP_DECLARE_UNARY(abs);
TF_CPP_DECLARE_UNARY(neg);

#undef TF_CPP_DECLARE_UNARY

template <typename T> auto pow(const nd_array<T> &a, T exponent) -> nd_array<T>;
template <typename T> auto pow_inplace(nd_array<T> &a, T exponent) -> void;
template <typename T>
auto atan2(const nd_array<T> &y, const nd_array<T> &x) -> nd_array<T>;
template <typename T>
auto clip(const nd_array<T> &a, T lower, T upper) -> nd_array<T>;
template <typename T>
auto clip_inplace(nd_array<T> &a, T lower, T upper) -> void;

#define TF_CPP_EXTERN_UNARY(NAME, T)                                           \
  extern template auto NAME<T>(const nd_array<T> &) -> nd_array<T>;            \
  extern template auto NAME##_inplace<T>(nd_array<T> &) -> void

#define TF_CPP_EXTERN_FLOAT_UNARIES(T)                                         \
  TF_CPP_EXTERN_UNARY(sqrt, T);                                                \
  TF_CPP_EXTERN_UNARY(sin, T);                                                 \
  TF_CPP_EXTERN_UNARY(cos, T);                                                 \
  TF_CPP_EXTERN_UNARY(tan, T);                                                 \
  TF_CPP_EXTERN_UNARY(asin, T);                                                \
  TF_CPP_EXTERN_UNARY(acos, T);                                                \
  TF_CPP_EXTERN_UNARY(atan, T);                                                \
  TF_CPP_EXTERN_UNARY(exp, T);                                                 \
  TF_CPP_EXTERN_UNARY(log, T);                                                 \
  TF_CPP_EXTERN_UNARY(log2, T);                                                \
  TF_CPP_EXTERN_UNARY(log10, T);                                               \
  TF_CPP_EXTERN_UNARY(floor, T);                                               \
  TF_CPP_EXTERN_UNARY(ceil, T);                                                \
  TF_CPP_EXTERN_UNARY(round, T);                                               \
  extern template auto pow<T>(const nd_array<T> &, T) -> nd_array<T>;          \
  extern template auto pow_inplace<T>(nd_array<T> &, T) -> void;               \
  extern template auto atan2<T>(const nd_array<T> &, const nd_array<T> &)      \
      -> nd_array<T>

TF_CPP_EXTERN_FLOAT_UNARIES(float);
TF_CPP_EXTERN_FLOAT_UNARIES(double);

#undef TF_CPP_EXTERN_FLOAT_UNARIES

#define TF_CPP_EXTERN_GENERAL(T)                                               \
  TF_CPP_EXTERN_UNARY(abs, T);                                                 \
  TF_CPP_EXTERN_UNARY(neg, T);                                                 \
  extern template auto clip<T>(const nd_array<T> &, T, T) -> nd_array<T>;      \
  extern template auto clip_inplace<T>(nd_array<T> &, T, T) -> void

TF_CPP_EXTERN_GENERAL(std::int8_t);
TF_CPP_EXTERN_GENERAL(std::int32_t);
TF_CPP_EXTERN_GENERAL(std::int64_t);
TF_CPP_EXTERN_GENERAL(float);
TF_CPP_EXTERN_GENERAL(double);

#undef TF_CPP_EXTERN_GENERAL
#undef TF_CPP_EXTERN_UNARY

} // namespace tf::cpp
