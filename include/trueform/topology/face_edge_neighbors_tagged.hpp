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
#include <array>
#include "./edge_id_in_face.hpp"
#include "./face_membership_like.hpp"

namespace tf {
namespace topology {
template <typename Policy, typename Index, typename Range, typename F>
auto face_edge_neighbors_tagged(const tf::face_membership_like<Policy> &blink,
                                const Range &faces, Index face_id,
                                const Index &v0, const Index &v1,
                                const F &apply) {
  const auto &range0 = blink[v0];
  auto it0 = range0.begin();
  auto end0 = range0.end();
  const auto &range1 = blink[v1];
  auto it1 = range1.begin();
  auto end1 = range1.end();
  // intersection on sorted ranges
  while ((it0 != end0) & (it1 != end1)) {
    if (*it0 > *it1)
      ++it0;
    else {
      if (char(Index(*it0) != face_id) & char(!(*it1 > *it0))) {
        const auto &face1 = faces[*it0];
        Index size = face1.size();
        Index edge_id = tf::edge_id_in_face(v0, v1, face1);
        if (edge_id != size &&
            apply(std::array<Index, 2>{Index(*it0++), edge_id}))
          return;
      }
      ++it1;
    }
  }
}
} // namespace topology

/// @ingroup topology_connectivity
/// @brief Applies a function to each face sharing an edge, with edge index.
///
/// Like @ref face_edge_neighbors_apply, but the callback receives
/// `std::array<Index, 2>{neighbor_face_id, edge_id_in_neighbor}` instead of
/// just the face ID.
///
/// No triangle specialization is used because the edge index in the neighbor
/// face must always be computed.
///
/// @tparam Index The integer type for indices.
/// @tparam Policy The face membership policy type.
/// @tparam Range The faces range type.
/// @tparam F The function type to apply.
/// @param blink The face membership structure.
/// @param faces The mesh faces.
/// @param face_id The face containing the edge.
/// @param v0 One endpoint of the edge.
/// @param v1 The other endpoint of the edge.
/// @param apply Function called with `std::array<Index,2>{face_id, edge_id}`.
///              Return true to stop.
template <typename Index, typename Policy, typename Range, typename F>
auto face_edge_neighbors_tagged_apply(
    const tf::face_membership_like<Policy> &blink, const Range &faces,
    Index face_id, const Index &v0, const Index &v1, const F &apply) {
  topology::face_edge_neighbors_tagged(blink, faces, face_id, v0, v1, apply);
}

/// @ingroup topology_connectivity
/// @brief Outputs tagged face-edge neighbor pairs to an iterator.
///
/// For a given edge (v0, v1) in a face, outputs
/// `std::array<Index, 2>{neighbor_face_id, edge_id_in_neighbor}` for each
/// neighboring face.
///
/// @tparam Index The integer type for indices.
/// @tparam Policy The face membership policy type.
/// @tparam Range The faces range type.
/// @tparam Iterator The output iterator type.
/// @param blink The face membership structure.
/// @param faces The mesh faces.
/// @param face_id The face containing the edge.
/// @param v0 One endpoint of the edge.
/// @param v1 The other endpoint of the edge.
/// @param out The output iterator.
/// @return The advanced output iterator.
template <typename Index, typename Policy, typename Range, typename Iterator>
auto face_edge_neighbors_tagged(const tf::face_membership_like<Policy> &blink,
                                const Range &faces, Index face_id,
                                const Index &v0, const Index &v1,
                                Iterator out) {
  topology::face_edge_neighbors_tagged(
      blink, faces, face_id, v0, v1, [&](const auto &val) {
        *out++ = val;
        return false;
      });
  return out;
}

/// @ingroup topology_connectivity
/// @brief Outputs tagged face-edge neighbor pairs to a bounded range.
///
/// For a given edge (v0, v1) in a face, outputs
/// `std::array<Index, 2>{neighbor_face_id, edge_id_in_neighbor}` for each
/// neighboring face, stopping when the output range is full.
///
/// @tparam Index The integer type for indices.
/// @tparam Policy The face membership policy type.
/// @tparam Range The faces range type.
/// @tparam Iterator The output iterator type.
/// @param blink The face membership structure.
/// @param faces The mesh faces.
/// @param face_id The face containing the edge.
/// @param v0 One endpoint of the edge.
/// @param v1 The other endpoint of the edge.
/// @param begin Start of output range.
/// @param end End of output range.
/// @return Iterator past the last written element.
template <typename Index, typename Policy, typename Range, typename Iterator>
auto face_edge_neighbors_tagged(const tf::face_membership_like<Policy> &blink,
                                const Range &faces, Index face_id,
                                const Index &v0, const Index &v1,
                                Iterator begin, Iterator end) {
  topology::face_edge_neighbors_tagged(
      blink, faces, face_id, v0, v1, [&](const auto &val) {
        *begin++ = val;
        return begin == end;
      });
  return begin;
}
} // namespace tf
