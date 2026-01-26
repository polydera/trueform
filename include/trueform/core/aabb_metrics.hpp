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
#include "./distance.hpp"
#include "./minimal_maximal_distance.hpp"

namespace tf {

/// @ingroup core_queries
/// @brief Distance metrics used for prioritized dual-tree traversal.
///
/// Contains two values:
/// - `min_d2`: the squared minimum distance between two AABBs.
/// - `min_max_d2`: a lower bound on the maximal pairwise distance between
///                 points from each AABB, useful for guiding dual-tree pruning.
///
/// Used internally during proximity queries (e.g., @ref tf::nearness_search).
/// Construct it using the helper function @ref tf::make_aabb_metrics
///
/// @tparam RealT The real-valued coordinate type (e.g., float or double).
template <typename RealT> struct aabb_metrics {
  /// @brief The squared minimum distance between two AABBs.
  RealT min_d2;
  /// @brief a lower bound on the maximal pairwise distance between
  ///          points from each AABB, useful for guiding dual-tree pruning.
  RealT min_max_d2;
};

/// @ingroup core_queries
/// @brief Compute distance metrics between two AABBs for use in dual-tree
/// queries.
///
/// Returns an @ref aabb_metrics<RealT> struct containing:
/// - `min_d2`: the squared minimum distance between `aabb0` and `aabb1`.
/// - `min_max_d2`: the minimal possible squared maximum distance between
///                 any pair of points, one from each AABB.
///
/// These values are used for guiding dual-tree traversal order and early
/// pruning.
///
/// @tparam RealT The real-valued coordinate type.
/// @tparam Dims The spatial dimension (typically 2 or 3).
/// @param aabb0 The first bounding box.
/// @param aabb1 The second bounding box.
/// @return aabb_metrics<RealT>
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_aabb_metrics(const aabb_like<Dims, Policy0> &aabb0,
                       const aabb_like<Dims, Policy1> &aabb1) {
  return aabb_metrics<tf::coordinate_type<Policy0, Policy1>>{
      tf::distance2(aabb0, aabb1), tf::minimal_maximal_distance2(aabb0, aabb1)};
}
} // namespace tf
