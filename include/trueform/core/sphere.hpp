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
#include "./base/sphere.hpp"
#include "./point_like.hpp"
#include "./sphere_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief A sphere in N-dimensional space.
///
/// Represents a sphere defined by its center (origin) and radius.
///
/// Use @ref tf::make_sphere() for construction.
///
/// @tparam T The scalar type (e.g., float, double).
/// @tparam Dims The dimensionality (e.g., 2, 3).
template <typename T, std::size_t Dims>
using sphere =
    tf::sphere_like<Dims, tf::core::sphere<Dims, tf::core::pt<T, Dims>>>;

/// @ingroup core_primitives
/// @brief Construct a sphere from a center point and radius.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The point policy type.
/// @param origin The center of the sphere.
/// @param r The radius.
/// @return A @ref tf::sphere instance.
template <std::size_t Dims, typename Policy>
auto make_sphere(const point_like<Dims, Policy> &origin,
                 tf::coordinate_type<Policy> r) {
  return sphere<tf::coordinate_type<Policy>, Dims>{origin, r};
}
} // namespace tf
