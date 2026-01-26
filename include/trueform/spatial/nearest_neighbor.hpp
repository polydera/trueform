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
#include "../core/metric_point.hpp"
#include "./tree_metric_info.hpp"

namespace tf {

/// @ingroup spatial_results
/// @brief Result type for single-tree nearest neighbor queries.
///
/// Alias for `tree_metric_info<Index, tf::metric_point<RealT, Dims>>`.
/// Contains the primitive ID, squared distance, and closest point.
///
/// @tparam Index The type used for primitive identifiers.
/// @tparam RealT The coordinate type (e.g., float or double).
/// @tparam Dims The spatial dimensionality.
template <typename Index, typename RealT, std::size_t Dims>
using nearest_neighbor =
    tree_metric_info<Index, tf::metric_point<RealT, Dims>>;
}
