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

/// @brief Return a row-sorted copy (value sort for one-dimensional arrays).
template <typename T> auto sort(const nd_array<T> &array) -> nd_array<T>;

/// @brief Row-sort an array in place (value sort for one-dimensional arrays).
template <typename T> auto sort_inplace(nd_array<T> &array) -> void;

/// @brief Return the lexicographic row permutation that sorts an array.
template <typename T>
auto argsort(const nd_array<T> &array) -> nd_array<std::int32_t>;

/// @brief Remove adjacent duplicate rows from a sorted array.
template <typename T> auto unique(const nd_array<T> &array) -> nd_array<T>;

/// @brief Merge two sorted row sets.
template <typename T>
auto set_union(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T>;

/// @brief Intersect two sorted row sets.
template <typename T>
auto set_intersection(const nd_array<T> &a, const nd_array<T> &b)
    -> nd_array<T>;

/// @brief Return rows in sorted array @p a that are not in sorted array @p b.
template <typename T>
auto set_difference(const nd_array<T> &a, const nd_array<T> &b) -> nd_array<T>;

#define TF_CPP_EXTERN_SORTING(T)                                               \
  extern template auto sort<T>(const nd_array<T> &) -> nd_array<T>;            \
  extern template auto sort_inplace<T>(nd_array<T> &) -> void;                 \
  extern template auto argsort<T>(const nd_array<T> &)                         \
      -> nd_array<std::int32_t>;                                               \
  extern template auto unique<T>(const nd_array<T> &) -> nd_array<T>;          \
  extern template auto set_union<T>(const nd_array<T> &, const nd_array<T> &)  \
      -> nd_array<T>;                                                          \
  extern template auto set_intersection<T>(                                    \
      const nd_array<T> &, const nd_array<T> &) -> nd_array<T>;                \
  extern template auto set_difference<T>(const nd_array<T> &,                  \
                                         const nd_array<T> &) -> nd_array<T>

TF_CPP_EXTERN_SORTING(std::int8_t);
TF_CPP_EXTERN_SORTING(std::int32_t);
TF_CPP_EXTERN_SORTING(float);
TF_CPP_EXTERN_SORTING(double);

#undef TF_CPP_EXTERN_SORTING

} // namespace tf::cpp
