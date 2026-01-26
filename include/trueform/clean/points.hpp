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
#include "../reindex/points.hpp"
#include "../reindex/return_index_map.hpp"
#include "./index_map/points.hpp"

namespace tf {

/// @ingroup clean
/// @brief Remove duplicate points with tolerance and return index map.
///
/// Removes duplicate points within the specified tolerance from a point
/// collection. Index type defaults to int (points have no index to deduce from).
///
/// @tparam Index The index type (defaults to int).
/// @tparam Policy The policy type of the points.
/// @param points The input @ref tf::points.
/// @param tolerance Points within this distance are considered duplicates.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Tuple of (@ref tf::points_buffer, @ref tf::index_map_buffer).
///
/// @see tf::make_clean_index_map for low-level index map generation.
/// @see tf::reindexed for applying the map to associated data.
template <typename Index = int, typename Policy>
auto cleaned(const tf::points<Policy> &points,
             tf::coordinate_type<Policy> tolerance, tf::return_index_map_t) {
  auto im = tf::make_clean_index_map<Index>(points, tolerance);
  auto out = tf::reindexed(points, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup clean
/// @brief Remove exact duplicate points and return index map.
/// @overload
template <typename Index = int, typename Policy>
auto cleaned(const tf::points<Policy> &points, tf::return_index_map_t) {
  auto im = tf::make_clean_index_map<Index>(points);
  auto out = tf::reindexed(points, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup clean
/// @brief Remove duplicate points within tolerance.
/// @overload
template <typename Index = int, typename Policy>
auto cleaned(const tf::points<Policy> &points,
             tf::coordinate_type<Policy> tolerance) {
  auto im = tf::make_clean_index_map<Index>(points, tolerance);
  auto out = tf::reindexed(points, im);
  return out;
}

/// @ingroup clean
/// @brief Remove exact duplicate points.
/// @overload
template <typename Index = int, typename Policy>
auto cleaned(const tf::points<Policy> &points) {
  auto im = tf::make_clean_index_map<Index>(points);
  auto out = tf::reindexed(points, im);
  return out;
}
} // namespace tf
