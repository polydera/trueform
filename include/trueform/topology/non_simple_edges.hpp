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
#include "../core/small_vector.hpp"
#include "../core/views/enumerate.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership.hpp"
#include "./policy/face_membership.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Computes both boundary and non-manifold edges in a single pass.
///
/// More efficient than calling @ref tf::make_boundary_edges() and
/// @ref tf::make_non_manifold_edges() separately, as it only traverses
/// the mesh once.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces of the mesh.
/// @param fm The face membership structure.
/// @return A pair of (boundary_edges, non_manifold_edges) as @ref tf::blocked_buffer.
template <typename Policy, typename Policy1>
auto make_non_simple_edges(const tf::faces<Policy> &faces,
                           const tf::face_membership_like<Policy1> &fm) {
  using Index = std::decay_t<decltype(fm[0][0])>;
  tf::blocked_buffer<Index, 2> boundary_edges;
  tf::blocked_buffer<Index, 2> non_manifold_edges;

  tf::generic_generate(
      tf::enumerate(faces),
      std::tie(boundary_edges.data_buffer(), non_manifold_edges.data_buffer()),
      tf::small_vector<Index, 10>{},
      [&](const auto &pair, auto &buffers, auto &neighbors) {
        auto &[boundary_buf, non_manifold_buf] = buffers;
        const auto &[face_id, face] = pair;
        Index size = face.size();
        Index prev = size - 1;
        for (Index i = 0; i < size; prev = i++) {
          neighbors.clear();
          tf::face_edge_neighbors(fm, faces, Index(face_id), Index(face[prev]),
                                  Index(face[i]), std::back_inserter(neighbors));
          if (neighbors.empty()) {
            // boundary edge: no neighboring faces
            boundary_buf.push_back(face[prev]);
            boundary_buf.push_back(face[i]);
          } else if (neighbors.size() > 1 &&
                     // only keep a single copy of an edge
                     std::all_of(neighbors.begin(), neighbors.end(),
                                 [fid = Index(face_id)](auto x) {
                                   return x > fid;
                                 })) {
            // non-manifold edge: more than one neighboring face
            non_manifold_buf.push_back(face[prev]);
            non_manifold_buf.push_back(face[i]);
          }
        }
      });

  return std::make_pair(std::move(boundary_edges), std::move(non_manifold_edges));
}

/// @ingroup topology_analysis
/// @brief Computes both boundary and non-manifold edges in a single pass.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons of the mesh.
/// @return A pair of (boundary_edges, non_manifold_edges) as @ref tf::blocked_buffer.
template <typename Policy>
auto make_non_simple_edges(const tf::polygons<Policy> &polygons) {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::make_non_simple_edges(polygons.faces(),
                                     polygons.face_membership());
  } else {
    tf::face_membership<std::decay_t<decltype(polygons.faces()[0][0])>> fm;
    fm.build(polygons);
    return tf::make_non_simple_edges(polygons.faces(), fm);
  }
}

} // namespace tf
