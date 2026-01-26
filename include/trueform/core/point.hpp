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

#include "./base/pt.hpp"
#include "./coordinate_type.hpp"
#include "./point_like.hpp"
#include "./vector_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Fixed-size N-dimensional point with element-wise arithmetic and
/// comparisons.
///
/// `tf::point<T, N>` is a general-purpose geometric point class supporting
/// standard point algebra (addition, subtraction, scalar multiplication, etc.)
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
/// Use `tf::make_point(...)` to construct points from raw arrays or pointers.
///
/// @tparam T The scalar element type (e.g., float, double, int).
/// @tparam N The dimensionality (e.g., 2, 3).
template <typename T, std::size_t N>
using point = tf::point_like<N, tf::core::pt<T, N>>;

/// @ingroup core_primitives
/// @brief Construct a point from a `std::array`.
///
/// Creates a @ref tf::point<T, N> by copying values from the given array.
///
/// @tparam T The scalar element type.
/// @tparam N The dimensionality.
/// @param arr The array to copy values from.
/// @return A `tf::point<T, N>` initialized from the array.
template <typename T, std::size_t N>
auto make_point(std::array<T, N> arr) -> point<T, N> {
  return point<T, N>(arr);
}

/// @ingroup core_primitives
/// @brief Construct a point from a raw pointer.
///
/// Creates a @ref tf::point<T, N> by copying `N` elements from the given
/// pointer. The pointer must reference a contiguous array of at least `N`
/// elements.
///
/// @tparam N The dimensionality.
/// @tparam T The scalar element type.
/// @param ptr Pointer to a contiguous block of `N` elements.
/// @return A `tf::point<T, N>` initialized from the pointer data.
template <std::size_t N, typename T>
auto make_point(const T *ptr) -> point<T, N> {
  return point<T, N>(ptr);
}

/// @ingroup core_primitives
/// @brief Construct a point from a vector.
///
/// Creates a @ref tf::point by copying coordinates from a vector-like type.
///
/// @tparam N The dimensionality.
/// @tparam T The vector policy type.
/// @param v The vector to convert.
/// @return A `tf::point` with the same coordinates.
template <typename T, std::size_t N>
auto make_point(const tf::vector_like<N, T> &v)
    -> point<tf::coordinate_type<T>, N> {
  point<tf::coordinate_type<T>, N> out;
  for (std::size_t i = 0; i < N; ++i)
    out[i] = v[i];
  return out;
}

/// @ingroup core_primitives
/// @brief Construct a point from individual coordinate values.
///
/// Creates a @ref tf::point by deducing type and dimensionality from
/// the provided arguments. Requires at least 2 coordinates.
///
/// @tparam T The coordinate type (deduced from first two arguments).
/// @tparam Ts Additional coordinate types.
/// @param t0 The first coordinate value.
/// @param t1 The second coordinate value.
/// @param ts Additional coordinate values.
/// @return A `tf::point<common_type, N>` where N = 2 + sizeof...(ts).
template <typename T, typename... Ts>
auto make_point(const T &t0, const T &t1, const Ts &...ts)
    -> tf::point<std::common_type_t<T, Ts...>, (2 + sizeof...(Ts))> {
  using type = std::common_type_t<T, Ts...>;
  return {static_cast<type>(t0), static_cast<type>(t1),
          static_cast<type>(ts)...};
}

} // namespace tf
