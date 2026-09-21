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

/// @brief One axis of a Cartesian multi-axis indexing operation.
class multi_take_index {
public:
  enum class mode { all, single, indices };

  multi_take_index() = default;
  explicit multi_take_index(int index);
  explicit multi_take_index(std::vector<std::int32_t> indices);

  auto selection_mode() const -> mode;
  auto single_index() const -> int;
  auto index_array() const -> const std::vector<std::int32_t> &;

private:
  mode _mode = mode::all;
  int _single_index = 0;
  std::vector<std::int32_t> _indices;
};

template <typename T>
auto take(const nd_array<T> &array, const nd_array<std::int32_t> &indices,
          int axis = 0) -> nd_array<T>;

template <typename T>
auto multi_take(const nd_array<T> &array,
                const std::vector<multi_take_index> &indices) -> nd_array<T>;

template <typename T>
auto take_along_axis(const nd_array<T> &array,
                     const nd_array<std::int32_t> &indices, int axis)
    -> nd_array<T>;

template <typename T>
auto boolean_index(const nd_array<T> &array, const nd_array<std::int8_t> &mask)
    -> nd_array<T>;

#define TF_CPP_EXTERN_INDEXING(T)                                              \
  extern template auto take<T>(const nd_array<T> &,                            \
                               const nd_array<std::int32_t> &, int)            \
      -> nd_array<T>;                                                          \
  extern template auto multi_take<T>(const nd_array<T> &,                      \
                                     const std::vector<multi_take_index> &)    \
      -> nd_array<T>;                                                          \
  extern template auto take_along_axis<T>(const nd_array<T> &,                 \
                                          const nd_array<std::int32_t> &, int) \
      -> nd_array<T>;                                                          \
  extern template auto boolean_index<T>(                                       \
      const nd_array<T> &, const nd_array<std::int8_t> &) -> nd_array<T>

TF_CPP_EXTERN_INDEXING(std::int8_t);
TF_CPP_EXTERN_INDEXING(std::int32_t);
TF_CPP_EXTERN_INDEXING(std::int64_t);
TF_CPP_EXTERN_INDEXING(float);
TF_CPP_EXTERN_INDEXING(double);

#undef TF_CPP_EXTERN_INDEXING

} // namespace tf::cpp
