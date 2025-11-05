/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./base/pt.hpp"
#include "./coordinate_type.hpp"
#include "./point_like.hpp"
#include "./vector_like.hpp"

namespace tf {

/// @ingroup geometry
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

/// @ingroup geometry
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

/// @ingroup geometry
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

template <typename T, std::size_t N>
auto make_point(const tf::vector_like<N, T> &v)
    -> point<tf::coordinate_type<T>, N> {
  point<tf::coordinate_type<T>, N> out;
  for (std::size_t i = 0; i < N; ++i)
    out[i] = v[i];
  return out;
}

} // namespace tf
