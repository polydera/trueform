/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./intersect_status.hpp"

namespace tf {
/// @ingroup geometry
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

/// @ingroup geometry
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

} // namespace tf
