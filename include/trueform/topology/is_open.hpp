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
#include "./is_closed.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Check if a mesh has boundary edges.
///
/// Returns `true` if any edge in the mesh is not shared by exactly two faces,
/// meaning the mesh has holes or open boundaries.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return `true` if the mesh is open (has boundary edges).
template <typename Policy, typename Policy1>
auto is_open(const tf::faces<Policy> &faces,
             const tf::face_membership_like<Policy1> &fm) -> bool {
  return !is_closed(faces, fm);
}

/// @ingroup topology_analysis
/// @brief Check if a mesh has boundary edges.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return `true` if the mesh is open (has boundary edges).
template <typename Policy>
auto is_open(const tf::polygons<Policy> &polygons) -> bool {
  return !is_closed(polygons);
}

/// @ingroup topology_analysis
/// @brief Check if a curve does not form a closed loop.
///
/// Returns `true` if the curve is non-empty and its first and last
/// points differ.
///
/// @tparam Dims The dimensionality of the curve points.
/// @tparam Policy The curve policy type.
/// @param curve The curve to check.
/// @return `true` if the curve is open.
template <std::size_t Dims, typename Policy>
auto is_open(const tf::curve<Dims, Policy> &curve) -> bool {
  return !is_closed(curve);
}

} // namespace tf
