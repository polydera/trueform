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
#include "../iter/value_iterator.hpp"
#include "../range.hpp"
namespace tf {

/// @ingroup core_ranges
/// @brief Creates a view over a half-open sequence of integral values [start,
/// end).
///
/// This function returns a lightweight, non-owning range of incrementing
/// values, similar to a `for (int i = start; i < end; ++i)` loop.
///
/// @tparam T The integral type of the sequence.
/// @param start The first value in the sequence.
/// @param end One past the last value in the sequence.
/// @return A view over the sequence [start, end).
///
/// @note This is similar to Python's `range(start, end)`.
/// @warning T must be an integral type; a static assertion is enforced.
template <typename T> auto make_sequence_range(T start, T end) {
  static_assert(std::is_integral_v<T>, "T must be integral.");
  auto begin_it = tf::iter::make_value(start);
  auto end_it = tf::iter::make_value(end);
  return tf::make_range(std::move(begin_it), std::move(end_it));
}

/// @ingroup core_ranges
/// @brief Creates a view over a half-open sequence of integral values [0, end).
///
/// This overload starts from zero and ends at the given value.
/// Equivalent to calling `make_sequence_range(0, end)`.
///
/// @tparam T The integral type of the sequence.
/// @param end One past the last value in the sequence.
/// @return A view over the sequence [0, end).
///
/// @note This is similar to Python's `range(end)`.
/// @warning T must be an integral type; a static assertion is enforced.
template <typename T> auto make_sequence_range(T end) {
  static_assert(std::is_integral_v<T>, "T must be integral.");
  auto begin_it = tf::iter::make_value(T(0));
  auto end_it = tf::iter::make_value(end);
  return tf::make_range(std::move(begin_it), std::move(end_it));
}

} // namespace tf
