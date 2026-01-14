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
#include "./dot.hpp"
#include "./sqrt.hpp"
#include "./vector_like.hpp"
namespace tf {

/// @ingroup core_queries
/// @brief Compute squared area of parallelogram spanned by two vectors.
///
/// Uses the identity |v0 x v1|^2 = |v0|^2 * |v1|^2 - (v0 . v1)^2.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam Policy0 The first vector's policy type.
/// @tparam Policy1 The second vector's policy type.
/// @param v0 First edge vector.
/// @param v1 Second edge vector.
/// @return The squared parallelogram area.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto parallelogram_area2(const vector_like<Dims, Policy0> &v0,
                         const vector_like<Dims, Policy1> &v1) {
  auto dot = tf::dot(v0, v1);
  return v0.length2() * v1.length2() - dot * dot;
}

/// @ingroup core_queries
/// @brief Compute area of parallelogram spanned by two vectors.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam Policy0 The first vector's policy type.
/// @tparam Policy1 The second vector's policy type.
/// @param v0 First edge vector.
/// @param v1 Second edge vector.
/// @return The parallelogram area.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto parallelogram_area(const vector_like<Dims, Policy0> &v0,
                        const vector_like<Dims, Policy1> &v1) {
  return tf::sqrt(parallelogram_area(v0, v1));
}

/// @ingroup core_queries
/// @brief Compute squared area of 2D parallelogram.
///
/// Optimized version using the 2D cross product (determinant).
///
/// @tparam Policy0 The first vector's policy type.
/// @tparam Policy1 The second vector's policy type.
/// @param v0 First edge vector.
/// @param v1 Second edge vector.
/// @return The squared parallelogram area.
template <typename Policy0, typename Policy1>
auto parallelogram_area2(const vector_like<2, Policy0> &v0,
                         const vector_like<2, Policy1> &v1) {
  auto tmp = v0[0] * v1[1] - v0[1] * v1[0];
  return tmp * tmp;
}

/// @ingroup core_queries
/// @brief Compute area of 2D parallelogram.
/// @overload
template <typename Policy0, typename Policy1>
auto parallelogram_area(const vector_like<2, Policy0> &v0,
                        const vector_like<2, Policy1> &v1) {
  auto tmp = v0[0] * v1[1] - v0[1] * v1[0];
  return std::abs(tmp);
}

/// @ingroup core_queries
/// @brief Compute signed area of 2D parallelogram.
///
/// Positive when v1 is counter-clockwise from v0.
///
/// @tparam Policy0 The first vector's policy type.
/// @tparam Policy1 The second vector's policy type.
/// @param v0 First edge vector.
/// @param v1 Second edge vector.
/// @return The signed parallelogram area (2D cross product).
template <typename Policy0, typename Policy1>
auto signed_parallelogram_area(const vector_like<2, Policy0> &v0,
                               const vector_like<2, Policy1> &v1) {
  return v0[0] * v1[1] - v0[1] * v1[0];
}
} // namespace tf
