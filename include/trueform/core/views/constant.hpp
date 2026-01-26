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
#include "./mapped_range.hpp"
#include "./sequence_range.hpp"

namespace tf {
namespace views {
template <typename T> struct constant_policy {
  T value;
  template <typename U> auto operator()(U &&) const { return value; }
};
} // namespace views
/// @ingroup core_ranges
/// @brief Creates a range that yields the same constant value `size` times.
///
/// Returns a view where every element is the same value. Useful for
/// broadcasting a single value across operations that expect a range.
///
/// @tparam T The type of the constant value.
/// @param value The constant value to repeat.
/// @param size The number of times to repeat the value.
/// @return A view yielding `value` for `size` iterations.
template <typename T> auto make_constant_range(T &&value, std::size_t size) {
  return tf::make_mapped_range(
      tf::make_sequence_range(size),
      views::constant_policy<std::decay_t<T>>{static_cast<T &&>(value)});
}
} // namespace tf
