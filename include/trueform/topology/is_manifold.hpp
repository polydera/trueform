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
#include "../core/checked.hpp"
#include "../core/faces.hpp"
#include "../core/polygons.hpp"
#include "../core/views/sequence_range.hpp"
#include "./face_membership_like.hpp"
#include "./make_face_membership.hpp"
#include "./policy/face_membership.hpp"
#include "./traversal/vertex_fan_is_manifold.hpp"
#include <type_traits>

namespace tf {

/// @ingroup topology_analysis
/// @brief Check if a mesh is manifold.
///
/// Returns `true` when the faces around every vertex are one fan, which also
/// says every edge carries at most two of them. Winding is a separate fact:
/// a mesh whose faces disagree about it is manifold all the same.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return `true` if the mesh is manifold.
template <typename Policy, typename Policy1>
auto is_manifold(const tf::faces<Policy> &faces,
                 const tf::face_membership_like<Policy1> &fm) -> bool {
  using Index = std::decay_t<decltype(faces[0][0])>;

  return !tf::parallel_contains(
      tf::make_sequence_range(Index(fm.size())),
      [&faces, &fm](Index v) {
        return !tf::topology::vertex_fan_is_manifold(faces, fm, v);
      },
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
/// @return `true` if the mesh is manifold.
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
