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

#include "./point_like.hpp"
#include "./coordinate_type.hpp"
#include "./vector_like.hpp"

namespace tf {
/// @ingroup core_properties
/// @brief Compute the dot product of two N-dimensional vectors.
///
/// Returns the standard inner product (a · b) between two vectors
/// of the same dimension.
///
/// @tparam N The dimensionality of the vectors.
/// @tparam T0 The vector policy
/// @tparam T1 The vector policy
/// @param a The first input vector.
/// @param b The second input vector.
/// @return The dot product of vectors `a` and `b`.
template <std::size_t N, typename T0, typename T1>
auto dot(const vector_like<N, T0> &a, const vector_like<N, T1> &b)
    -> tf::coordinate_type<T0, T1> {
  tf::coordinate_type<T0, T1> sum{0};
  for (std::size_t i = 0; i < N; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

/// @ingroup core_properties
/// @brief Compute the dot product treating a point as a position vector.
template <std::size_t N, typename T0, typename T1>
auto dot(const point_like<N, T0> &a, const vector_like<N, T1> &b)
    -> tf::coordinate_type<T0, T1> {
  return dot(a.as_vector_view(), b);
}

/// @ingroup core_properties
/// @brief Compute the dot product treating a point as a position vector.
template <std::size_t N, typename T0, typename T1>
auto dot(const vector_like<N, T0> &a, const point_like<N, T1> &b)
    -> tf::coordinate_type<T0, T1> {
  return dot(a, b.as_vector_view());
}
} // namespace tf
