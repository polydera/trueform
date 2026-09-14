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
#include <vector>

namespace tf::cpp {

template <typename T>
auto stack(const std::vector<nd_array<T>> &arrays, int axis) -> nd_array<T>;

template <typename T>
auto concatenate(const std::vector<nd_array<T>> &arrays, int axis)
    -> nd_array<T>;

template <typename T>
auto tile(const nd_array<T> &array, const std::vector<int> &repetitions)
    -> nd_array<T>;

template <typename T> auto transpose(const nd_array<T> &array) -> nd_array<T>;

template <typename T>
auto transpose(const nd_array<T> &array, const std::vector<int> &axes)
    -> nd_array<T>;

template <typename T>
auto where(const nd_array<std::int8_t> &condition, const nd_array<T> &x,
           const nd_array<T> &y) -> nd_array<T>;

template <typename T> auto clone(const nd_array<T> &array) -> nd_array<T>;

#define TF_CPP_EXTERN_ND_ARRAY_STRUCTURE(T)                                    \
  extern template auto stack<T>(const std::vector<nd_array<T>> &, int)         \
      -> nd_array<T>;                                                          \
  extern template auto concatenate<T>(const std::vector<nd_array<T>> &, int)   \
      -> nd_array<T>;                                                          \
  extern template auto tile<T>(const nd_array<T> &, const std::vector<int> &)  \
      -> nd_array<T>;                                                          \
  extern template auto transpose<T>(const nd_array<T> &) -> nd_array<T>;       \
  extern template auto transpose<T>(const nd_array<T> &,                       \
                                    const std::vector<int> &) -> nd_array<T>;  \
  extern template auto where<T>(const nd_array<std::int8_t> &,                 \
                                const nd_array<T> &, const nd_array<T> &)      \
      -> nd_array<T>;                                                          \
  extern template auto clone<T>(const nd_array<T> &) -> nd_array<T>

TF_CPP_EXTERN_ND_ARRAY_STRUCTURE(std::int8_t);
TF_CPP_EXTERN_ND_ARRAY_STRUCTURE(std::int32_t);
TF_CPP_EXTERN_ND_ARRAY_STRUCTURE(float);
TF_CPP_EXTERN_ND_ARRAY_STRUCTURE(double);

#undef TF_CPP_EXTERN_ND_ARRAY_STRUCTURE

} // namespace tf::cpp
