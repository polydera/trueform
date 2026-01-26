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

#include "./vector_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Fixed-size N-dimensional vector with element-wise arithmetic and
/// comparisons.
///
/// `tf::vector<T, N>` is a general-purpose geometric vector class supporting
/// standard vector algebra (addition, subtraction, scalar multiplication, etc.)
/// and comparisons. It is used throughout the library for spatial coordinates,
/// offsets, and directions.
///
/// The class supports:
/// - Element access via `operator[]`
/// - Component-wise arithmetic (`+`, `-`, `*`, `/`)
/// - Comparisons (`==`, `!=`, `<`, `>` if defined for T)
/// - Iteration and pointer conversion
/// - conversions using as<U>() method
/// - length() method
///
/// Use `tf::make_vector(...)` to construct vectors from raw arrays or pointers.
///
/// @tparam T The scalar element type (e.g., float, double, int).
/// @tparam N The dimensionality (e.g., 2, 3).
template <typename T, std::size_t N>
using vector = tf::vector_like<N, core::vec<T, N>>;

/// @ingroup core_primitives
/// @brief Construct a vector from a `std::array`.
///
/// Creates a @ref tf::vector<T, N> by copying values from the given array.
///
/// @tparam T The scalar element type.
/// @tparam N The dimensionality.
/// @param arr The array to copy values from.
/// @return A `tf::vector<T, N>` initialized from the array.
template <typename T, std::size_t N>
auto make_vector(std::array<T, N> arr) -> vector<T, N> {
  return vector<T, N>(arr);
}

/// @ingroup core_primitives
/// @brief Construct a vector from a raw pointer.
///
/// Creates a @ref tf::vector<T, N> by copying `N` elements from the given
/// pointer. The pointer must reference a contiguous array of at least `N`
/// elements.
///
/// @tparam N The dimensionality.
/// @tparam T The scalar element type.
/// @param ptr Pointer to a contiguous block of `N` elements.
/// @return A `tf::vector<T, N>` initialized from the pointer data.
template <std::size_t N, typename T>
auto make_vector(const T *ptr) -> vector<T, N> {
  return vector<T, N>(ptr);
}

/// @ingroup core_primitives
/// @brief Construct a vector from individual coordinate values.
///
/// Creates a @ref tf::vector by deducing type and dimensionality from
/// the provided arguments. Requires at least 2 coordinates.
///
/// @tparam T The coordinate type (deduced from first two arguments).
/// @tparam Ts Additional coordinate types.
/// @param t0 The first coordinate value.
/// @param t1 The second coordinate value.
/// @param ts Additional coordinate values.
/// @return A `tf::vector<common_type, N>` where N = 2 + sizeof...(ts).
template <typename T, typename... Ts>
auto make_vector(const T &t0, const T &t1, const Ts &...ts)
    -> tf::vector<std::common_type_t<T, Ts...>, (2 + sizeof...(Ts))> {
  using type = std::common_type_t<T, Ts...>;
  return {static_cast<type>(t0), static_cast<type>(t1),
          static_cast<type>(ts)...};
}

} // namespace tf
