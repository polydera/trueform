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
#include "../core/metric_point_pair.hpp"
#include "./tree_metric_info_pair.hpp"

namespace tf {

/// @ingroup spatial_results
/// @brief Result type for dual-tree nearest neighbor queries.
///
/// Alias for `tree_metric_info_pair<Index0, Index1, tf::metric_point_pair<RealT, Dims>>`.
/// Contains primitive IDs from both trees, squared distance, and closest points on each.
///
/// @tparam Index0 The index type for the first tree.
/// @tparam Index1 The index type for the second tree.
/// @tparam RealT The coordinate type (e.g., float or double).
/// @tparam Dims The spatial dimensionality.
template <typename Index0, typename Index1, typename RealT, std::size_t Dims>
using nearest_neighbor_pair =
    tree_metric_info_pair<Index0, Index1, tf::metric_point_pair<RealT, Dims>>;
}
