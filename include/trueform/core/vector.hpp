/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./vector_like.hpp"

namespace tf {

/// @ingroup geometry
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

/// @ingroup geometry
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

/// @ingroup geometry
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

} // namespace tf
