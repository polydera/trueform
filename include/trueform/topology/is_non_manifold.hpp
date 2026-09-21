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
#include "./is_manifold.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Check if a mesh is not manifold.
///
/// The negation of @ref tf::is_manifold.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return `true` if the mesh is not manifold.
template <typename Policy, typename Policy1>
auto is_non_manifold(const tf::faces<Policy> &faces,
                     const tf::face_membership_like<Policy1> &fm) -> bool {
  return !is_manifold(faces, fm);
}

/// @ingroup topology_analysis
/// @brief Check if a mesh is not manifold.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return `true` if the mesh is not manifold.
template <typename Policy>
auto is_non_manifold(const tf::polygons<Policy> &polygons) -> bool {
  return !is_manifold(polygons);
}

} // namespace tf
