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

#define TF_CPP_DECLARE_COMPARISON(NAME)                                        \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &a, const nd_array<T> &b)                        \
      -> nd_array<std::int8_t>;                                                \
  template <typename T>                                                        \
  auto NAME##_scalar(const nd_array<T> &a, T scalar) -> nd_array<std::int8_t>

TF_CPP_DECLARE_COMPARISON(eq);
TF_CPP_DECLARE_COMPARISON(neq);
TF_CPP_DECLARE_COMPARISON(lt);
TF_CPP_DECLARE_COMPARISON(gt);
TF_CPP_DECLARE_COMPARISON(lte);
TF_CPP_DECLARE_COMPARISON(gte);

#undef TF_CPP_DECLARE_COMPARISON

#define TF_CPP_EXTERN_COMPARISON(NAME, T)                                      \
  extern template auto NAME<T>(const nd_array<T> &, const nd_array<T> &)       \
      -> nd_array<std::int8_t>;                                                \
  extern template auto NAME##_scalar<T>(const nd_array<T> &, T)                \
      -> nd_array<std::int8_t>

#define TF_CPP_EXTERN_COMPARISONS(T)                                           \
  TF_CPP_EXTERN_COMPARISON(eq, T);                                             \
  TF_CPP_EXTERN_COMPARISON(neq, T);                                            \
  TF_CPP_EXTERN_COMPARISON(lt, T);                                             \
  TF_CPP_EXTERN_COMPARISON(gt, T);                                             \
  TF_CPP_EXTERN_COMPARISON(lte, T);                                            \
  TF_CPP_EXTERN_COMPARISON(gte, T)

TF_CPP_EXTERN_COMPARISONS(std::int8_t);
TF_CPP_EXTERN_COMPARISONS(std::int32_t);
TF_CPP_EXTERN_COMPARISONS(float);
TF_CPP_EXTERN_COMPARISONS(double);

#undef TF_CPP_EXTERN_COMPARISONS
#undef TF_CPP_EXTERN_COMPARISON

} // namespace tf::cpp
