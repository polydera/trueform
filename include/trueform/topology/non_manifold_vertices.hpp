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
#include "../core/algorithm/sequenced_generate.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/faces.hpp"
#include "../core/polygons.hpp"
#include "../core/views/sequence_range.hpp"
#include "./face_membership.hpp"
#include "./face_membership_like.hpp"
#include "./policy/face_membership.hpp"
#include "./traversal/vertex_fan_is_manifold.hpp"
#include <type_traits>

namespace tf {

/// @ingroup topology_analysis
/// @brief Extract the non-manifold vertices from faces and face membership.
///
/// A vertex is non-manifold when the faces around it are not one fan: an edge
/// at it carries three or more faces, or its faces fall into several pieces
/// that meet at the vertex alone. Winding is another fact and never enters,
/// and a vertex no face names is not reported.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return A @ref tf::buffer of the non-manifold vertex ids, ascending.
template <typename Policy, typename Policy1>
auto make_non_manifold_vertices(const tf::faces<Policy> &faces,
                                const tf::face_membership_like<Policy1> &fm) {
  using Index = std::decay_t<decltype(faces[0][0])>;
  tf::buffer<Index> vertices;
  tf::sequenced_generate(
      tf::make_sequence_range(Index(fm.size())), vertices,
      [&faces, &fm](Index v, tf::buffer<Index> &buffer) {
        if (!tf::topology::vertex_fan_is_manifold(faces, fm, v))
          buffer.push_back(v);
      },
      tf::checked);
  return vertices;
}

/// @ingroup topology_analysis
/// @brief Extract the non-manifold vertices from a polygons range.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return A @ref tf::buffer of the non-manifold vertex ids, ascending.
template <typename Policy>
auto make_non_manifold_vertices(const tf::polygons<Policy> &polygons) {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::make_non_manifold_vertices(polygons.faces(),
                                          polygons.face_membership());
  } else {
    tf::face_membership<std::decay_t<decltype(polygons.faces()[0][0])>> fe;
    fe.build(polygons);
    return tf::make_non_manifold_vertices(polygons.faces(), fe);
  }
}

} // namespace tf
