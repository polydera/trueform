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
#include "./intersect_status.hpp"

namespace tf {
/// @ingroup core_queries
/// @brief Lightweight result of a ray casting query.
///
/// Contains only the intersection status and the parametric hit distance `t`,
/// without computing or storing the actual intersection point.
///
/// @tparam RealT The scalar type used for the parametric distance.
template <typename RealT> struct ray_cast_info {
  using real_t = RealT;
  /// Status of the ray cast (e.g., intersection, none, parallel, error).
  tf::intersect_status status = tf::intersect_status::none;

  /// Parametric distance `t` along the ray direction at which the intersection
  /// occurs. Meaningful only if `status == intersect_status::intersection`.
  RealT t;

  /// @brief Checks if the ray intersects the object.
  /// @return `true` if `status == intersect_status::intersection`, `false`
  /// otherwise.
  operator bool() const { return status == tf::intersect_status::intersection; }
};

/// @ingroup core_queries
/// @brief Helper to construct a `ray_cast_info` result.
///
/// @tparam RealT The scalar type used for the parametric distance.
/// @param status The intersection status.
/// @param t The parametric distance along the ray.
/// @return A `ray_cast_info<RealT>` instance.
template <typename RealT>
auto make_ray_cast_info(tf::intersect_status status, RealT t) {
  return ray_cast_info<RealT>{status, t};
}

namespace core {
template <typename RealT>
auto does_intersect_any(tf::ray_cast_info<RealT> info) {
  return info.status == tf::intersect_status::intersection ||
         info.status == tf::intersect_status::colinear ||
         info.status == tf::intersect_status::coplanar;
}
} // namespace core

} // namespace tf
