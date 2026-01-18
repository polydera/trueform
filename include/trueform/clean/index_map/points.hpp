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
#include "../../core/algorithm/make_equivalence_class_index_map.hpp"
#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/distance.hpp"
#include "../../core/index_map.hpp"
#include "../../core/points.hpp"
#include "../../spatial/aabb_tree.hpp"
#include "../../spatial/gather_self_ids.hpp"
#include "../../spatial/policy/tree.hpp"

namespace tf {

/// @ingroup clean
/// @brief Generate index map for point deduplication (output parameter).
///
/// Creates an @ref tf::index_map_buffer mapping old point indices to new.
/// Use @ref tf::reindexed to apply the map to associated data.
///
/// @tparam Policy The policy type of the points.
/// @tparam Index The index type.
/// @param points The input @ref tf::points.
/// @param im Output @ref tf::index_map_buffer to populate.
template <typename Policy, typename Index>
auto make_clean_index_map(const tf::points<Policy> &points,
                          tf::index_map_buffer<Index> &im) {
  if (!points.size())
    return;
  tf::make_unique_index_map(points, im);
}

/// @ingroup clean
/// @brief Generate index map for point deduplication with tolerance (output parameter).
/// @overload
template <typename Policy, typename Index>
auto make_clean_index_map(const tf::points<Policy> &points,
                          tf::coordinate_type<Policy> tolerance,
                          tf::index_map_buffer<Index> &im) {
  if (!points.size())
    return;
  if (tolerance == 0)
    return make_clean_index_map(points, im);
  if constexpr (tf::has_tree_policy<Policy>) {
    tf::buffer<std::array<Index, 2>> ids;
    tf::gather_self_ids(
        points,
        [d2 = tolerance * tolerance](const auto &x, const auto &y) {
          return distance2(x, y) <= d2;
        },
        std::back_inserter(ids));
    tf::make_dense_equivalence_class_index_map(ids, points.size(), im);
  } else {
    constexpr std::size_t dims = tf::static_size_v<decltype(points[0])>;
    tf::aabb_tree<Index, tf::coordinate_type<Policy>, dims> tree;
    tree.build(points, tf::config_tree(4, 4));
    return make_clean_index_map(points | tf::tag(tree), tolerance, im);
  }
}

/// @ingroup clean
/// @brief Generate index map for point deduplication with tolerance.
///
/// Creates an @ref tf::index_map_buffer mapping old point indices to new.
/// Use @ref tf::reindexed to apply the map to associated data.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Policy The policy type of the points.
/// @param points The input @ref tf::points.
/// @param tolerance Points within this distance are considered duplicates.
/// @return An @ref tf::index_map_buffer for the points.
template <typename Index = int, typename Policy>
auto make_clean_index_map(const tf::points<Policy> &points,
                          tf::coordinate_type<Policy> tolerance) {
  tf::index_map_buffer<Index> point_map;
  make_clean_index_map(points, tolerance, point_map);
  return point_map;
}

/// @ingroup clean
/// @brief Generate index map for exact point deduplication.
/// @overload
template <typename Index = int, typename Policy>
auto make_clean_index_map(const tf::points<Policy> &points) {
  tf::index_map_buffer<Index> point_map;
  make_clean_index_map(points, point_map);
  return point_map;
}
} // namespace tf
