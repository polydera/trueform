/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <limits>
namespace tf {
/// @ingroup geometry
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

/// @ingroup geometry
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
