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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/area.hpp"
#include "../core/buffer.hpp"
#include "../core/faces.hpp"
#include "../core/polygons.hpp"
#include "../core/views/constant.hpp"
#include "../core/views/indirect_range.hpp"
#include "./directed_edge_id_in_face.hpp"
#include "./face_membership.hpp"
#include "./manifold_edge_link.hpp"
#include "./policy/manifold_edge_link.hpp"

namespace tf {
namespace topology {

/// @ingroup topology_analysis
/// @brief Orient faces consistently using weighted voting.
///
/// Traverses the mesh by connected components and ensures all faces within
/// each component have consistent winding. Uses weights (typically face area)
/// to decide which orientation to keep when a region needs to be flipped.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The manifold edge link policy type.
/// @tparam Range The weights range type.
/// @param faces The faces range (modified in place).
/// @param link The manifold edge link structure.
/// @param weights Per-face weights for voting (e.g., face areas).
template <typename Policy, typename Policy1, typename Range>
void orient_faces_consistently(tf::faces<Policy> &faces,
                               const tf::manifold_edge_link_like<Policy1> &link,
                               const Range &weights) {
  using Index = std::decay_t<decltype(link[0][0].face_peer)>;
  using Weight = std::decay_t<decltype(weights[0])>;

  tf::buffer<bool> visited;
  visited.allocate(faces.size());
  tf::parallel_fill(visited, false);

  tf::buffer<Index> queue;
  queue.reserve(64);

  for (Index seed = 0; seed < static_cast<Index>(faces.size()); ++seed) {
    if (visited[seed])
      continue;

    // Start new region
    visited[seed] = true;
    queue.clear();
    queue.push_back(seed);

    Weight weight_flipped = 0;
    Weight weight_not_flipped = weights[seed]; // seed is not flipped

    std::size_t queue_begin = 0;

    while (queue_begin < queue.size()) {
      Index curr = queue[queue_begin++];

      const auto &curr_face = faces[curr];
      const auto &curr_link = link[curr];

      Index size = curr_link.size();
      Index prev = size - 1;
      for (Index i = 0; i < size; prev = i++) {
        const auto &peer = curr_link[prev];
        if (!peer.is_simple())
          continue;

        Index neighbor = peer.face_peer;
        if (visited[neighbor])
          continue;

        auto &&neighbor_face = faces[neighbor];
        Index a = curr_face[prev];
        Index b = curr_face[i];

        // Check direction in neighbor face
        Index dir = tf::directed_edge_id_in_face(b, a, neighbor_face);
        if (dir == Index(neighbor_face.size())) {
          // Edge direction reversed → flip neighbor face
          std::reverse(neighbor_face.begin(), neighbor_face.end());
          weight_flipped += weights[neighbor];
        } else {
          weight_not_flipped += weights[neighbor];
        }
        visited[neighbor] = true;
        queue.push_back(neighbor);
      }
    }

    // If flipped faces have more weight, flip entire region
    if (weight_flipped > weight_not_flipped) {
      tf::parallel_for_each(
          tf::make_indirect_range(queue, faces),
          [](auto &&face) { std::reverse(face.begin(), face.end()); });
    }
  }
}
} // namespace topology

/// @ingroup topology_analysis
/// @brief Orient faces consistently with uniform weights.
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The manifold edge link policy type.
/// @param faces The faces range (modified in place).
/// @param link The manifold edge link structure.
template <typename Policy, typename Policy1>
void orient_faces_consistently(
    tf::faces<Policy> &faces,
    const tf::manifold_edge_link_like<Policy1> &link) {
  return topology::orient_faces_consistently(
      faces, link, tf::make_constant_range(1, faces.size()));
}

/// @ingroup topology_analysis
/// @overload
template <typename Policy, typename Policy1>
void orient_faces_consistently(
    tf::faces<Policy> &&faces,
    const tf::manifold_edge_link_like<Policy1> &link) {
  return tf::orient_faces_consistently(faces, link);
}

/// @ingroup topology_analysis
/// @brief Orient faces consistently in a polygons range.
///
/// Uses face area as weights for voting. Builds manifold edge link internally
/// if not provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range (modified in place).
template <typename Policy>
void orient_faces_consistently(tf::polygons<Policy> &polygons) {
  auto weights = tf::make_mapped_range(
      polygons, [](const auto &poly) { return tf::area2(poly); });
  if constexpr (tf::has_manifold_edge_link_policy<Policy>)
    return topology::orient_faces_consistently(
        polygons.faces(), polygons.manifold_edge_link(), weights);
  else {
    using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
    tf::face_membership<Index> fm;
    fm.build(polygons);
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(polygons.faces()[0])>>
        mel;
    mel.build(polygons.faces(), fm);
    return topology::orient_faces_consistently(polygons.faces(), mel, weights);
  }
}

/// @ingroup topology_analysis
/// @overload
template <typename Policy>
void orient_faces_consistently(tf::polygons<Policy> &&polygons) {
  return orient_faces_consistently(polygons);
}
} // namespace tf
