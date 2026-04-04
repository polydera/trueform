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
#include "../core/angle.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/dot.hpp"
#include "../core/policy/normals.hpp"
#include "../core/polygons.hpp"
#include "../core/views/enumerate.hpp"
#include "../topology/make_manifold_edge_link.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./compute_normals.hpp"

namespace tf {

/// @ingroup geometry
/// @brief Extract sharp edges from a polygon mesh.
///
/// Returns edges where the dihedral angle between adjacent face normals
/// exceeds the given threshold. Boundary and non-manifold edges are excluded.
///
/// Uses manifold edge link if available via policy, otherwise computes it.
/// Uses normals if available via policy, otherwise computes them.
///
/// @param polygons The input polygons (must be 3D).
/// @param angle_threshold Minimum dihedral angle for an edge to be sharp.
/// @return A blocked_buffer of vertex index pairs for feature edges.
template <typename Policy>
auto make_sharp_edges(const tf::polygons<Policy> &polygons,
                      tf::rad<tf::coordinate_type<Policy>> angle_threshold) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;

  auto cos_threshold = tf::cos(angle_threshold);

  // Get or compute manifold edge link
  auto run = [&](const auto &mel, const auto &normals) {
    tf::blocked_buffer<Index, 2> edges;
    tf::generic_generate(tf::enumerate(polygons.faces()), edges.data_buffer(),
                         [&](const auto &pair, auto &buffer) {
                           const auto &[face_id, face] = pair;
                           Index size = face.size();
                           Index prev = size - 1;
                           for (Index i = 0; i < size; prev = i++) {
                             auto &&peer = mel[face_id][prev];
                             if (!peer.is_simple())
                               continue;
                             if (!peer.is_representative(Index(face_id)))
                               continue;
                             auto d = tf::dot(normals[face_id],
                                              normals[peer.face_peer]);
                             if (d < cos_threshold) {
                               buffer.push_back(std::min(face[prev], face[i]));
                               buffer.push_back(std::max(face[prev], face[i]));
                             }
                           }
                         });
    return edges;
  };

  auto dispatch_normals = [&](const auto &mel) {
    if constexpr (tf::has_normals_policy<Policy>) {
      return run(mel, polygons.normals());
    } else {
      auto normals = tf::compute_normals(polygons);
      return run(mel, normals.unit_vectors());
    }
  };

  if constexpr (tf::has_manifold_edge_link_policy<Policy>) {
    return dispatch_normals(polygons.manifold_edge_link());
  } else {
    auto mel = tf::make_manifold_edge_link(polygons);
    return dispatch_normals(mel);
  }
}

/// @overload
template <typename Policy>
auto make_sharp_edges(const tf::polygons<Policy> &polygons,
                      tf::deg<tf::coordinate_type<Policy>> angle_threshold) {
  return make_sharp_edges(
      polygons, tf::rad<tf::coordinate_type<Policy>>{angle_threshold});
}

} // namespace tf
