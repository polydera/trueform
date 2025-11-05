/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./aabb_like.hpp"
#include "./base/aabb.hpp"
#include "./coordinate_type.hpp"

namespace tf {

/// @ingroup geometry
/// @brief Axis-aligned bounding box in N-dimensional space.
///
/// Represents a rectangular region defined by its component-wise `min` and
/// `max` corners. Used extensively for spatial indexing, proximity queries, and
/// partitioning operations.
///
/// @tparam T The scalar type of the coordinates (e.g., float or double).
/// @tparam N The spatial dimension (e.g., 2 or 3).
template <typename T, std::size_t Dims>
using aabb = tf::aabb_like<
    Dims, tf::core::aabb<Dims, tf::core::pt<T, Dims>, tf::core::pt<T, Dims>>>;

/// @ingroup geometry
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
