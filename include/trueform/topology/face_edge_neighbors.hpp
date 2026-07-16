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
#include "../core/static_size.hpp"
#include "./edge_id_in_face.hpp"
#include "./face_membership_like.hpp"

namespace tf {
namespace topology {
template <typename Policy, typename Index, typename Range, typename F>
auto face_edge_neighbors(std::integral_constant<std::size_t, 3>,
                         const tf::face_membership_like<Policy> &blink,
                         const Range &, Index face_id, const Index &v0,
                         const Index &v1, const F &apply) {

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
        if (apply(*it0++))
          return;
      }
      ++it1;
    }
  }
}
template <std::size_t N, typename Policy, typename Index, typename Range,
          typename F>
auto face_edge_neighbors(std::integral_constant<std::size_t, N>,
                         const tf::face_membership_like<Policy> &blink,
                         const Range &faces, Index face_id, const Index &v0,
                         const Index &v1, const F &apply) {

  const auto &range0 = blink[v0];
  auto it0 = range0.begin();
  auto end0 = range0.end();
  const auto &range1 = blink[v1];
  auto it1 = range1.begin();
  auto end1 = range1.end();
  // Intersection on sorted ranges. A pinched face (a vertex it visits
  // twice) appears twice in both memberships, so the same candidate can
  // match repeatedly. The emission multiplicity must equal the number
  // of times the face actually holds the edge: a slit traversing the
  // edge twice is a neighbour twice, a lobe pinch that merely revisits
  // the endpoints is a neighbour once. Duplicate hits are adjacent in
  // the sorted ranges, so the full recount runs only when one shows.
  Index prev = face_id;
  Index emitted = 0;
  Index count = 0;
  bool full_count = false;
  while ((it0 != end0) & (it1 != end1)) {
    if (*it0 > *it1)
      ++it0;
    else {
      if (char(Index(*it0) != face_id) & char(!(*it1 > *it0))) {
        Index cand = Index(*it0);
        const auto &face1 = faces[cand];
        Index size = Index(face1.size());
        if (cand != prev) {
          prev = cand;
          emitted = 0;
          full_count = false;
          count =
              Index(tf::edge_id_in_face(v0, v1, face1) != std::size_t(size));
        } else if (!full_count) {
          full_count = true;
          Index c = 0;
          Index p = size - 1;
          for (Index i = 0; i < size; p = i++)
            c += Index((char(face1[p] == v0) & char(face1[i] == v1)) |
                       (char(face1[p] == v1) & char(face1[i] == v0)));
          count = c;
        }
        if (emitted < count) {
          ++emitted;
          if (apply(cand))
            return;
        }
        ++it0;
      }
      ++it1;
    }
  }
}
} // namespace topology

/// @ingroup topology_connectivity
/// @brief Applies a function to each face sharing an edge.
///
/// For a given edge (v0, v1) in a face, finds all other faces that share
/// this edge and applies the given function to their face IDs.
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
/// @param apply Function called with each neighbor face ID. Return true to stop.
template <typename Index,typename Policy, typename Range, typename F>
auto face_edge_neighbors_apply(const tf::face_membership_like<Policy> &blink,
                               const Range &faces, Index face_id,
                               const Index &v0, const Index &v1,
                               const F &apply) {
  topology::face_edge_neighbors(
      std::integral_constant<std::size_t,
                             tf::static_size_v<decltype(faces[face_id])>>{},
      blink, faces, face_id, v0, v1, apply);
}

/// @ingroup topology_connectivity
/// @brief Outputs all faces sharing an edge to an iterator.
///
/// For a given edge (v0, v1) in a face, outputs all other face IDs that
/// share this edge.
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
auto face_edge_neighbors(const tf::face_membership_like<Policy> &blink,
                         const Range &faces, Index face_id, const Index &v0,
                         const Index &v1, Iterator out) {
  topology::face_edge_neighbors(
      std::integral_constant<std::size_t,
                             tf::static_size_v<decltype(faces[face_id])>>{},
      blink, faces, face_id, v0, v1, [&](const auto &val) {
        *out++ = val;
        return false;
      });
  return out;
}

/// @ingroup topology_connectivity
/// @brief Outputs faces sharing an edge to a bounded range.
///
/// For a given edge (v0, v1) in a face, outputs other face IDs that share
/// this edge, stopping when the output range is full.
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
auto face_edge_neighbors(const tf::face_membership_like<Policy> &blink,
                         const Range &faces, Index face_id, const Index &v0,
                         const Index &v1, Iterator begin, Iterator end) {
  topology::face_edge_neighbors(
      std::integral_constant<std::size_t,
                             tf::static_size_v<decltype(faces[face_id])>>{},
      blink, faces, face_id, v0, v1, [&](const auto &val) {
        *begin++ = val;
        return begin == end;
      });
  return begin;
}
} // namespace tf
