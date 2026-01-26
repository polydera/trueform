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
#include <limits>
namespace tf {
/// @ingroup core_queries
/// @brief Configuration for ray casting operations.
///
/// Defines the parametric bounds `[min_t, max_t]` along a ray
/// within which intersections will be considered.
///
/// @tparam RealT The floating-point type used for parametric bounds.
template <typename RealT> struct ray_config {
  /// Lower bound along the ray direction to consider (inclusive).
  RealT min_t = 0;

  /// Upper bound along the ray direction to consider (inclusive).
  RealT max_t = std::numeric_limits<RealT>::max();
};

/// @ingroup core_queries
/// @brief Creates a configured `ray_cast_config` object with the given bounds.
///
/// This is a convenience helper to construct a `ray_cast_config`
/// for use in bounded ray casting queries.
///
/// @tparam RealT The floating-point type used for parametric bounds.
/// @param min_t The minimum parametric value along the ray.
/// @param max_t The maximum parametric value along the ray.
/// @return A `ray_cast_config<RealT>` with the specified bounds.
///
/// @see ray_cast_config
template <typename RealT> auto make_ray_config(RealT min_t, RealT max_t) {
  return ray_config<RealT>{min_t, max_t};
}

} // namespace tf
