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
#include "./sqrt.hpp"

namespace tf {

/// @ingroup core_queries
/// @brief Computes the squared maximal distance between two corners of two
/// AABBs.
///
/// This function returns the maximum squared Euclidean distance between the
/// corresponding corners (`min` and `max`) of two axis-aligned bounding boxes
/// (AABBs). It is useful as a cheap, conservative approximation of the furthest
/// possible distance between points inside the boxes.
///
/// Specifically, it computes:
/// \code
/// max(length2(a.min - b.min), length2(a.max - b.max))
/// \endcode
///
/// @tparam T Numeric type.
/// @tparam N Dimensionality of the AABB (e.g., 2D, 3D).
/// @param a First AABB.
/// @param b Second AABB.
/// @return Maximum squared distance between corresponding corners of `a` and
/// `b`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto minimal_maximal_distance2(const aabb_like<Dims, Policy0> &a,
                               const aabb_like<Dims, Policy1> &b) {
  return std::max((a.min - b.min).length2(), (a.max - b.max).length2());
}

/// @ingroup core_queries
/// @brief Computes the maximal distance between two corners of two AABBs.
///
/// This function returns the square root of the result of
/// @ref minimal_maximal_distance2. It is an upper bound approximation of the
/// maximal distance between any points inside AABBs `a` and `b`.
///
/// @tparam T Numeric type.
/// @tparam N Dimensionality of the AABB (e.g., 2D, 3D).
/// @param a First AABB.
/// @param b Second AABB.
/// @return Maximum Euclidean distance between corners of `a` and `b`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto minimal_maximal_distance(const aabb_like<Dims, Policy0> &a,
                              const aabb_like<Dims, Policy1> &b) {
  return tf::sqrt(minimal_maximal_distance2(a, b));
}

} // namespace tf
