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
#include "../../core/algorithm/circular_decrement.hpp"
#include "../../core/algorithm/circular_increment.hpp"
#include "../edge_id_in_face.hpp"
#include "../face_edge_neighbors.hpp"
#include "../face_membership_like.hpp"

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief Where a fan walk stands: a corner of a face at the vertex, and
/// which of the corner's two edges the walk leaves by.
///
/// @tparam Index The integer type for indices.
template <typename Index> struct vertex_fan_position {
  Index face;
  Index corner;
  bool leave_by_next;
};

/// @ingroup topology_analysis
/// @brief Step the fan walk over the edge the corner does not enter through.
///
/// @tparam Index The integer type for indices.
/// @tparam Faces The faces range type.
/// @tparam Policy The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @param v The vertex.
/// @param at The position, moved onto the peer's corner when it is crossed.
/// @return The number of faces on that edge beside this one, counted to two;
///   `at` moves exactly when that number is one.
template <typename Index, typename Faces, typename Policy>
auto vertex_fan_step(const Faces &faces,
                     const tf::face_membership_like<Policy> &fm, const Index &v,
                     vertex_fan_position<Index> &at) -> Index {
  const auto &face = faces[at.face];
  Index n = Index(face.size());
  Index w = at.leave_by_next
                ? Index(face[tf::circular_increment(at.corner, n)])
                : Index(face[tf::circular_decrement(at.corner, n)]);
  Index peer = -1;
  Index peers = 0;
  tf::face_edge_neighbors_apply(fm, faces, at.face, v, w,
                                [&peer, &peers](const auto &id) {
                                  peer = Index(id);
                                  return ++peers > 1;
                                });
  if (peers != 1)
    return peers;
  const auto &peer_face = faces[peer];
  Index e = Index(tf::edge_id_in_face(v, w, peer_face));
  Index j = Index(peer_face[e]) == v
                ? e
                : tf::circular_increment(e, Index(peer_face.size()));
  at = vertex_fan_position<Index>{peer, j, j != e};
  return peers;
}

} // namespace tf::topology
