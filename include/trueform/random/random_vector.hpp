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

#include "../core/vector.hpp"
#include "./random.hpp"

namespace tf {

/// @ingroup random
/// @brief Generates a random vector of dimension `N` with values in the range
/// [`from`, `to`].
///
/// Each component of the resulting vector is independently sampled using
/// `tf::random(from, to)`. Note: This overload requires that `N` is a known
/// compile-time constant.
///
/// @tparam N Dimension of the vector.
/// @tparam T Numeric type of the vector components.
/// @param from The lower bound (inclusive) for each component.
/// @param to The upper bound (exclusive) for each component.
/// @return A `tf::vector<T, N>` filled with random values in [`from`, `to`].
///
/// @see tf::random
template <int N, typename T>
auto random_vector(T from, T to) -> tf::vector<T, N> {
  return tf::vector<T, N>{tf::random(from, to), tf::random(from, to),
                          tf::random(from, to)};
}

/// @ingroup random
/// @brief Generates a random vector of dimension `N` with values in the range
/// [0, 1).
///
/// This is a convenience overload that calls `tf::random_vector<N>(T(0),
/// T(1))`.
///
/// @tparam T Numeric type of the vector components.
/// @tparam N Dimension of the vector.
/// @return A `tf::vector<T, N>` filled with random values in [0, 1).
///
/// @see tf::random_vector
template <typename T, std::size_t N> auto random_vector() -> tf::vector<T, N> {
  return tf::random_vector<N>(T(0), T(1));
}
} // namespace tf
