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
#include "./base/plane.hpp"
#include "./dot.hpp"
#include "./normal.hpp"
#include "./plane_like.hpp"
#include "./point_like.hpp"
#include "./unit_vector_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief An infinite plane in N-dimensional space.
///
/// A plane is defined by a unit normal vector and a signed distance `d` from
/// the origin. Points on the plane satisfy: `dot(normal, point) + d = 0`.
///
/// Factory functions include:
/// - @ref tf::make_plane() - from normal+offset, normal+point, or three points
/// - @ref tf::make_plane_like() - creates view when inputs are views
///
/// @tparam T The scalar type (e.g., float, double).
/// @tparam Dims The dimensionality (e.g., 3).
template <typename T, std::size_t Dims>
using plane =
    tf::plane_like<Dims, tf::core::plane<Dims, tf::core::vec<T, Dims>>>;

/// @ingroup core_primitives
/// @brief Constructs a plane from a normal and offset.
///
/// Assumes the given normal is already normalized. This is a low-level overload
/// useful when the offset `d` is precomputed or known.
///
/// @tparam T Scalar type.
/// @tparam N Number of dimensions.
/// @param normal A unit-length normal vector.
/// @param d Signed offset from the origin.
/// @return A `plane<T, N>` representing the given plane.
template <std::size_t N, typename T>
auto make_plane(const unit_vector_like<N, T> &normal,
                tf::coordinate_type<T> d) {
  return plane<tf::coordinate_type<T>, N>{normal, d};
}

/// @ingroup core_primitives
/// @brief Constructs a plane from a normal vector and a point on the plane.
///
/// Computes the plane offset from the point and the unit normal vector:
/// `d = -dot(normal, point)`.
///
/// @tparam T0 Scalar type of the unit normal.
/// @tparam N Number of dimensions.
/// @tparam T1 A vector-like type convertible to a point.
/// @param normal A unit-length normal vector.
/// @param point A point on the plane.
/// @return A `plane<T0, N>` that passes through the given point.
template <std::size_t N, typename T0, typename T1>
auto make_plane(const unit_vector_like<N, T0> &normal,
                const point_like<N, T1> &point) {
  auto val = -tf::dot(normal, point);
  return plane<decltype(val), N>{normal, val};
}

/// @ingroup core_primitives
/// @brief Constructs a plane from three points.
///
/// Computes the normal vector via the cross product (or generalization in N-D),
/// then calculates the offset so the plane passes through the first point.
///
/// @tparam Dims Number of dimensions.
/// @tparam T0, T1, T2 Vector-like input types for the three points.
/// @param pt0 First point (used to compute the offset).
/// @param pt1 Second point (used for the normal).
/// @param pt2 Third point (used for the normal).
/// @return A `plane` defined by the three points.
template <std::size_t Dims, typename T0, typename T1, typename T2>
auto make_plane(const point_like<Dims, T0> &pt0,
                const point_like<Dims, T1> &pt1,
                const point_like<Dims, T2> &pt2) {
  auto normal = tf::make_normal(pt0, pt1, pt2);
  auto val = -tf::dot(normal, pt0);
  return tf::plane<decltype(val), Dims>{normal, val};
}
} // namespace tf
