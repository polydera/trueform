/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./unit_vector.hpp"
#include "./value_type.hpp"
#include "./vector_like.hpp"

namespace tf {
/// @ingroup geometry
/// @brief Return a normalized copy of a vector view.
///
/// Creates a copy of the input @ref tf::vector_view, normalizes it using @ref
/// normalize(), and returns the result. The original input remains unchanged.
///
/// @tparam T The scalar type (e.g., float or double).
/// @tparam Dims The dimensionality of the vector.
/// @param v The input vector view to normalize.
/// @return A normalized vector of type @ref tf::vector<T, Dims>.
template <std::size_t N, typename T>
auto normalized(const vector_like<N, T> &v)
    -> tf::unit_vector<tf::value_type<T>, N> {
  return v;
}
} // namespace tf
