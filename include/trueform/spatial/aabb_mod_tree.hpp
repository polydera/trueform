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

#include "../core/aabb.hpp"
#include "./mod_tree.hpp"

namespace tf {

/// @ingroup spatial_structures
/// @brief A dynamic spatial tree using axis-aligned bounding boxes (AABBs).
///
/// Alias for `tf::mod_tree<Index, tf::aabb<RealT, Dims>>`.
///
/// @tparam Index The type used for primitive identifiers.
/// @tparam RealT The real-valued coordinate type (e.g., float or double).
/// @tparam Dims The spatial dimension (typically 2 or 3).
template <typename Index, typename RealT, std::size_t Dims>
using aabb_mod_tree = tf::mod_tree<Index, tf::aabb<RealT, Dims>>;

} // namespace tf
