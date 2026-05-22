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
#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/coordinate_dims.hpp"
#include "../../core/index_map.hpp"
#include "../../core/points.hpp"
#include "../../exact/resolve_int_type.hpp"
#include "../../exact/snap.hpp"
#include <array>

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
  using Real = tf::coordinate_type<Policy>;
  if (tolerance == 0)
    return make_clean_index_map(points, im);

  using SnapInt = tf::exact::resolve_int_type<tf::none_t, Real>;
  auto snap = [tolerance](const auto &p) {
    constexpr std::size_t Dims = tf::coordinate_dims_v<Policy>;
    std::array<SnapInt, Dims> q;
    for (std::size_t d = 0; d < Dims; ++d)
      q[d] = static_cast<SnapInt>(tf::exact::snap_key(Real(p[d]), tolerance));
    return q;
  };
  auto eq = [&](const auto &a, const auto &b) { return snap(a) == snap(b); };
  auto less = [&](const auto &a, const auto &b) { return snap(a) < snap(b); };
  tf::make_unique_index_map(points, im, eq, less);
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
