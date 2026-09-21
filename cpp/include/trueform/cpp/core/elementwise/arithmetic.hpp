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

#define TF_CPP_DECLARE_BINARY(NAME)                                            \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T>;        \
  template <typename T>                                                        \
  auto NAME##_inplace(nd_array<T> &a, const nd_array<T> &b) -> void;           \
  template <typename T>                                                        \
  auto NAME##_scalar(const nd_array<T> &a, T scalar) -> nd_array<T>;           \
  template <typename T>                                                        \
  auto NAME##_scalar_inplace(nd_array<T> &a, T scalar) -> void

TF_CPP_DECLARE_BINARY(add);
TF_CPP_DECLARE_BINARY(sub);
TF_CPP_DECLARE_BINARY(mul);
TF_CPP_DECLARE_BINARY(div);
TF_CPP_DECLARE_BINARY(mod);

#undef TF_CPP_DECLARE_BINARY

#define TF_CPP_EXTERN_BINARY(NAME, T)                                          \
  extern template auto NAME<T>(const nd_array<T> &, const nd_array<T> &)       \
      -> nd_array<T>;                                                          \
  extern template auto NAME##_inplace<T>(nd_array<T> &, const nd_array<T> &)   \
      -> void;                                                                 \
  extern template auto NAME##_scalar<T>(const nd_array<T> &, T)                \
      -> nd_array<T>;                                                          \
  extern template auto NAME##_scalar_inplace<T>(nd_array<T> &, T) -> void

#define TF_CPP_EXTERN_ARITHMETIC(T)                                            \
  TF_CPP_EXTERN_BINARY(add, T);                                                \
  TF_CPP_EXTERN_BINARY(sub, T);                                                \
  TF_CPP_EXTERN_BINARY(mul, T);                                                \
  TF_CPP_EXTERN_BINARY(div, T);                                                \
  TF_CPP_EXTERN_BINARY(mod, T)

TF_CPP_EXTERN_ARITHMETIC(std::int8_t);
TF_CPP_EXTERN_ARITHMETIC(std::int32_t);
TF_CPP_EXTERN_ARITHMETIC(std::int64_t);
TF_CPP_EXTERN_ARITHMETIC(float);
TF_CPP_EXTERN_ARITHMETIC(double);

#undef TF_CPP_EXTERN_ARITHMETIC
#undef TF_CPP_EXTERN_BINARY

} // namespace tf::cpp
