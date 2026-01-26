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
#include "./base/ray.hpp"
#include "./ray_like.hpp"
#include "./coordinate_type.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief A ray originating from a point and extending infinitely in one
/// direction.
///
/// A ray has an origin point and a direction vector. Unlike a line, it only
/// extends in the positive direction from the origin. Supports parametric
/// evaluation: `ray(t)` returns `origin + t * direction`.
///
/// Factory functions include:
/// - @ref tf::make_ray() - from origin and direction (always owning)
/// - @ref tf::make_ray_between_points() - from two points
///
/// @tparam T The scalar type (e.g., float, double).
/// @tparam Dims The dimensionality (e.g., 2, 3).
template <typename T, std::size_t Dims>
using ray = tf::ray_like<
    Dims, tf::core::ray<Dims, tf::core::pt<T, Dims>, tf::core::vec<T, Dims>>>;

/// @ingroup core_primitives
/// @brief Constructs a ray from an origin and a direction vector.
///
/// The direction does not need to be normalized. The resulting ray's
/// value type is the common type between the two vector components.
///
/// @tparam T0 The type of the origin vector's components.
/// @tparam T1 The type of the direction vector's components.
/// @tparam Dims The spatial dimension of the vectors.
/// @param origin The starting point of the ray.
/// @param direction The direction vector of the ray.
/// @return A ray object.
template <typename T0, std::size_t Dims, typename T1>
auto make_ray(const tf::point_like<Dims, T0> &origin,
              const tf::vector_like<Dims, T1> &direction) {
  return ray<tf::coordinate_type<T0, T1>, Dims>{origin, direction};
}

/// @ingroup core_primitives
/// @brief Constructs a ray from two points, with the direction pointing from
/// origin to end.
///
/// The direction is computed as `end - origin`. The resulting ray's
/// value type is the common type between the two points.
///
/// @tparam T0 The type of the origin point.
/// @tparam T1 The type of the end point.
/// @tparam Dims The spatial dimension of the points.
/// @param origin The starting point of the ray.
/// @param end The end point; the direction of the ray will point from `origin`
/// to `end`.
/// @return A ray object.
template <typename T0, std::size_t Dims, typename T1>
auto make_ray_between_points(const tf::point_like<Dims, T0> &origin,
                             const tf::point_like<Dims, T1> &end) {
  return ray<tf::coordinate_type<T0, T1>, Dims>{origin, end - origin};
}

} // namespace tf
