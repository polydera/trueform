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

#include "./aabb_like.hpp"
#include "./base/aabb.hpp"
#include "./coordinate_type.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Axis-aligned bounding box in N-dimensional space.
///
/// Represents a rectangular region defined by its component-wise `min` and
/// `max` corners. Used extensively for spatial indexing, proximity queries, and
/// partitioning operations.
///
/// Factory functions include:
/// - @ref tf::make_aabb() - from min/max corners
/// - @ref tf::make_aabb_like() - view when inputs are views
/// - @ref tf::aabb_from() - compute AABB of any finite primitive
///
/// @tparam T The scalar type of the coordinates (e.g., float or double).
/// @tparam N The spatial dimension (e.g., 2 or 3).
template <typename T, std::size_t Dims>
using aabb = tf::aabb_like<
    Dims, tf::core::aabb<Dims, tf::core::pt<T, Dims>, tf::core::pt<T, Dims>>>;

/// @ingroup core_primitives
/// @brief Construct an AABB from `min` and `max` corners.
///
/// A convenience function equivalent to directly calling the `aabb<T, N>`
/// constructor.
///
/// @tparam N The spatial dimension.
/// @tparam T0 The vector policy
/// @tparam T1 The vector policy
/// @param min The lower corner of the bounding box.
/// @param max The upper corner of the bounding box.
/// @return An `aabb<T, N>` instance.
template <std::size_t N, typename T0, typename T1>
auto make_aabb(const point_like<N, T0> &min, const point_like<N, T1> &max)
    -> aabb<tf::coordinate_type<T0, T1>, N> {
  return aabb<tf::coordinate_type<T0, T1>, N>(min, max);
}

} // namespace tf
