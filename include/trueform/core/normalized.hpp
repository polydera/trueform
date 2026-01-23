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
#include "./unit_vector.hpp"
#include "./coordinate_type.hpp"
#include "./vector_like.hpp"

namespace tf {
/// @ingroup core_primitives
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
    -> tf::unit_vector<tf::coordinate_type<T>, N> {
  return v;
}

/// @ingroup core_primitives
/// @brief Return a copy of a unit vector (already normalized).
template <std::size_t N, typename T>
auto normalized(const unit_vector_like<N, T> &v)
    -> tf::unit_vector<tf::coordinate_type<T>, N> {
  return v;
}
} // namespace tf
