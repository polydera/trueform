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
#include "../core/algorithm/generic_generate.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/faces.hpp"
#include "../core/views/enumerate.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership.hpp"
#include "./policy/face_membership.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Extract boundary edges from faces and face membership.
///
/// Returns all edges that belong to only one face (boundary edges).
/// These are edges on the mesh boundary where the surface has a hole
/// or open edge.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return A @ref tf::blocked_buffer containing pairs of vertex indices for boundary edges.
template <typename Policy, typename Policy1>
auto make_boundary_edges(const tf::faces<Policy> &faces,
                         const tf::face_membership_like<Policy1> &fm) {
  using Index = std::decay_t<decltype(fm[0][0])>;
  tf::blocked_buffer<Index, 2> edges;
  tf::generic_generate(tf::enumerate(faces), edges.data_buffer(),
                       [&](const auto &pair, auto &buffer) {
                         const auto &[face_id, face] = pair;
                         Index size = face.size();
                         Index prev = size - 1;
                         std::array<Index, 1> neighbors;
                         for (Index i = 0; i < size; prev = i++) {
                           auto it = tf::face_edge_neighbors(
                               fm, faces, Index(face_id), Index(face[prev]),
                               Index(face[i]), neighbors.begin(),
                               neighbors.end());
                           if (it == neighbors.begin()) {
                             buffer.push_back(face[prev]);
                             buffer.push_back(face[i]);
                           }
                         }
                       });
  return edges;
}

/// @ingroup topology_analysis
/// @brief Extract boundary edges from a polygons range.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return A @ref tf::blocked_buffer containing pairs of vertex indices for boundary edges.
template <typename Policy>
auto make_boundary_edges(const tf::polygons<Policy> &polygons) {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::make_boundary_edges(polygons.faces(),
                                   polygons.face_membership());
  } else {
    tf::face_membership<std::decay_t<decltype(polygons.faces()[0][0])>> fe;
    fe.build(polygons);
    return tf::make_boundary_edges(polygons.faces(), fe);
  }
}
} // namespace tf
