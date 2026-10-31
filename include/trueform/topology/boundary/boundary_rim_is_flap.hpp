/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/faces.hpp"
#include "../boundary_rims.hpp"
#include "../face_membership_like.hpp"
#include "../traversal/vertex_carries_non_manifold_edge.hpp"
#include <cstddef>

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief Whether a boundary rim is a flap rather than a hole.
///
/// Every boundary rim is exactly one of the two, decided by its ends. A
/// closed rim bounds missing skin and is a hole; a sheet's free rim reads
/// the same, the answer standing against the expectation of a closed
/// solid. An open rim is a flap when each of its ends carries an edge three
/// or more faces share, the attachment its faces hang from, and a hole
/// otherwise: a pinch of the boundary alone hangs nothing.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @tparam Index The integer type for vertex and face indices.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @param rims The mesh's @ref tf::boundary_rims.
/// @param rim The rim.
/// @return `true` if the rim is a flap, `false` if it is a hole.
template <typename Policy, typename Policy1, typename Index>
auto boundary_rim_is_flap(const tf::faces<Policy> &faces,
                          const tf::face_membership_like<Policy1> &fm,
                          const tf::boundary_rims<Index> &rims, std::size_t rim)
    -> bool {
  if (rims.closed[rim])
    return false;
  const auto vertices = rims.vertices[rim];
  return tf::topology::vertex_carries_non_manifold_edge(faces, fm,
                                                        vertices.front()) &&
         tf::topology::vertex_carries_non_manifold_edge(faces, fm,
                                                        vertices.back());
}

} // namespace tf::topology
