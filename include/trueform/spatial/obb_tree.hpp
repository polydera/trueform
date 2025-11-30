/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/obb.hpp"
#include "./tree.hpp"

namespace tf {

/// @ingroup spatial_structures
/// @brief A spatial tree using oriented bounding boxes (OBBs).
///
/// Alias for `tf::tree<Index, tf::obb<RealT, Dims>>`.
///
/// @tparam Index The type used for primitive identifiers.
/// @tparam RealT The real-valued coordinate type (e.g., float or double).
/// @tparam Dims The spatial dimension (typically 2 or 3).
template <typename Index, typename RealT, std::size_t Dims>
using obb_tree = tf::tree<Index, tf::obb<RealT, Dims>>;

} // namespace tf
