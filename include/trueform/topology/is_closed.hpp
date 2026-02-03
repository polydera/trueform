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
#include "../core/curve.hpp"
#include "../core/views/enumerate.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership_like.hpp"
#include "./make_face_membership.hpp"
#include "./policy/face_membership.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Check if a mesh has no boundary edges.
///
/// Returns `true` if every edge in the mesh is shared by at least one other
/// face, meaning the mesh has no holes or open boundaries. A closed mesh is
/// watertight and encloses a volume.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return `true` if the mesh is closed (no boundary edges).
template <typename Policy, typename Policy1>
auto is_closed(const tf::faces<Policy> &faces,
               const tf::face_membership_like<Policy1> &fm) -> bool {
  using Index = std::decay_t<decltype(faces[0][0])>;

  auto has_boundary_edge = [&](auto pair) {
    auto [face_id, face] = pair;
    auto size = face.size();
    decltype(size) prev = size - 1;
    for (decltype(size) i = 0; i < size; prev = i++) {
      auto v0 = face[prev];
      auto v1 = face[i];
      bool found = false;
      tf::face_edge_neighbors_apply(fm, faces, Index(face_id), Index(v0),
                                    Index(v1), [&](auto) {
                                      found = true;
                                      return true;
                                    });
      if (!found)
        return true;
    }
    return false;
  };

  return !tf::parallel_contains(tf::enumerate(faces), has_boundary_edge,
                                tf::checked);
}

/// @ingroup topology_analysis
/// @brief Check if a mesh has no boundary edges.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return `true` if the mesh is closed (no boundary edges).
template <typename Policy>
auto is_closed(const tf::polygons<Policy> &polygons) -> bool {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::is_closed(polygons.faces(), polygons.face_membership());
  } else {
    auto fm = tf::make_face_membership(polygons);
    return is_closed(polygons | tf::tag(fm));
  }
}

/// @ingroup topology_analysis
/// @brief Check if a curve forms a closed loop.
///
/// Returns `true` if the curve is empty or if its first and last
/// points are the same, forming a closed loop.
///
/// @tparam Dims The dimensionality of the curve points.
/// @tparam Policy The curve policy type.
/// @param curve The curve to check.
/// @return `true` if the curve is closed.
template <std::size_t Dims, typename Policy>
auto is_closed(const tf::curve<Dims, Policy> &curve) -> bool {
  return !curve.size() || curve.front() == curve.back();
}
} // namespace tf
