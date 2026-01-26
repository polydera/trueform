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
#include "./point.hpp"
#include "./coordinate_type.hpp"

namespace tf {

/// @ingroup core_queries
/// @brief Result of a nearest-point query between two primitives or trees.
///
/// Represents the closest pair of points—one from each object or tree—along
/// with the distance metric (typically squared distance). Used in dual-tree
/// nearest neighbor queries such as @ref tf::nearness_search(tree0, tree1). Use
/// `tf::make_closest_point_pair` to create an instance.
///
/// @tparam RealT The scalar coordinate type (e.g., float or double).
/// @tparam Dims The spatial dimension (e.g., 2 or 3).
template <typename RealT, std::size_t Dims> struct metric_point_pair {
  /// @brief The metric between the points
  RealT metric;
  /// @brief The first point
  point<RealT, Dims> first;
  /// @brief The second point
  point<RealT, Dims> second;
};

/// @ingroup core_queries
/// @brief Construct a `closest_point_pair` from a distance and two spatial
/// points.
///
/// Convenience function to create a `closest_point_pair<RealT, Dims>` instance
/// without explicitly specifying the type.
///
/// @tparam RealT The scalar coordinate type.
/// @tparam Dims The spatial dimension.
/// @tparam T0 The vector policy
/// @tparam T1 The vector policy
/// @param metric The distance metric (typically squared distance).
/// @param first The closest point from the first object or tree.
/// @param second The closest point from the second object or tree.
/// @return A `closest_point_pair<RealT, Dims>` instance.
template <typename RealT, std::size_t Dims, typename T0, typename T1>
auto make_metric_point_pair(RealT metric, point_like<Dims, T0> first,
                            point_like<Dims, T1> second) {
  return metric_point_pair<tf::coordinate_type<RealT, T0, T1>, Dims>{metric, first,
                                                                  second};
}

template <typename RealT, std::size_t Dims>
auto min(const metric_point_pair<RealT, Dims> &lhs,
         const metric_point_pair<RealT, Dims> &rhs)
    -> const metric_point_pair<RealT, Dims> & {
  if (lhs.metric < rhs.metric)
    return lhs;
  return rhs;
}

template <typename RealT, std::size_t Dims>
auto max(const metric_point_pair<RealT, Dims> &lhs,
         const metric_point_pair<RealT, Dims> &rhs)
    -> const metric_point_pair<RealT, Dims> & {
  if (lhs.metric > rhs.metric)
    return lhs;
  return rhs;
}
} // namespace tf
