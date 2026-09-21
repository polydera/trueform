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
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/faces.hpp"
#include "../../core/views/enumerate.hpp"
#include "../face_edge_neighbors.hpp"
#include "../face_membership_like.hpp"
#include <array>
#include <tuple>

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief The mesh's boundary: every edge one face alone carries, beside
///        that face.
///
/// The two are one fact and stay aligned: the edge at block `k` of `edges`
/// is carried by `edge_faces[k]` and by nothing else. A face states its
/// edges in corner order and the faces come in their own, so the boundary
/// is the mesh's order rather than the schedule's.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @tparam Index The integer type for vertex and face indices.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @param edges Appended the two vertex ids of every boundary edge.
/// @param edge_faces Appended the face carrying each of those edges.
template <typename Policy, typename Policy1, typename Index>
auto gather_boundary_edges(const tf::faces<Policy> &faces,
                           const tf::face_membership_like<Policy1> &fm,
                           tf::buffer<Index> &edges,
                           tf::buffer<Index> &edge_faces) -> void {
  tf::sequenced_generate(
      tf::enumerate(faces), std::tie(edges, edge_faces),
      [&faces, &fm](const auto &pair, auto &buffers) {
        auto &[edge_buffer, face_buffer] = buffers;
        const auto &[face_id, face] = pair;
        Index size = face.size();
        Index prev = size - 1;
        std::array<Index, 1> neighbors;
        for (Index i = 0; i < size; prev = i++) {
          auto it = tf::face_edge_neighbors(fm, faces, Index(face_id),
                                            Index(face[prev]), Index(face[i]),
                                            neighbors.begin(), neighbors.end());
          if (it == neighbors.begin()) {
            edge_buffer.push_back(face[prev]);
            edge_buffer.push_back(face[i]);
            face_buffer.push_back(Index(face_id));
          }
        }
      });
}

} // namespace tf::topology
