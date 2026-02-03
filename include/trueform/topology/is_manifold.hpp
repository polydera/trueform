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
#include "../core/algorithm/parallel_contains.hpp"
#include "../core/views/enumerate.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership_like.hpp"
#include "./make_face_membership.hpp"
#include "./policy/face_membership.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Check if a mesh is manifold.
///
/// Returns `true` if every edge in the mesh is shared by at most two faces.
/// Non-manifold edges (shared by 3+ faces) indicate self-intersections or
/// invalid topology.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return `true` if the mesh is manifold (no edges shared by 3+ faces).
template <typename Policy, typename Policy1>
auto is_manifold(const tf::faces<Policy> &faces,
                 const tf::face_membership_like<Policy1> &fm) -> bool {
  using Index = std::decay_t<decltype(faces[0][0])>;

  auto has_non_manifold_edge = [&](auto pair) {
    auto [face_id, face] = pair;
    auto size = face.size();
    decltype(size) prev = size - 1;
    for (decltype(size) i = 0; i < size; prev = i++) {
      auto v0 = face[prev];
      auto v1 = face[i];
      int count = 0;
      tf::face_edge_neighbors_apply(fm, faces, Index(face_id), Index(v0),
                                    Index(v1), [&](auto) {
                                      ++count;
                                      return count > 1;
                                    });
      if (count > 1)
        return true;
    }
    return false;
  };

  return !tf::parallel_contains(tf::enumerate(faces), has_non_manifold_edge,
                                tf::checked);
}

/// @ingroup topology_analysis
/// @brief Check if a mesh is manifold.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return `true` if the mesh is manifold (no edges shared by 3+ faces).
template <typename Policy>
auto is_manifold(const tf::polygons<Policy> &polygons) -> bool {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::is_manifold(polygons.faces(), polygons.face_membership());
  } else {
    auto fm = tf::make_face_membership(polygons);
    return is_manifold(polygons | tf::tag(fm));
  }
}

} // namespace tf
