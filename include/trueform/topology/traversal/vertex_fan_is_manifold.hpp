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
#include "../face_membership_like.hpp"
#include "./vertex_fan_step.hpp"

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief Whether the corners at a vertex walk as one fan.
///
/// The walk leaves a corner by the edge it did not enter through and steps
/// onto the face on that edge's other side: a third face on an edge ends it,
/// and so does a corner the walk never reaches. A vertex no face names has
/// nothing to break.
///
/// @tparam Index The integer type for indices.
/// @tparam Faces The faces range type.
/// @tparam Policy The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @param v The vertex.
/// @return `true` if the fan is one open or closed cycle of corners.
template <typename Index, typename Faces, typename Policy>
auto vertex_fan_is_manifold(const Faces &faces,
                            const tf::face_membership_like<Policy> &fm,
                            const Index &v) -> bool {
  Index corners = 0;
  Index seed_face = -1;
  Index seed_corner = 0;
  Index previous_face = -1;
  for (const auto &id : fm[v]) {
    auto f = Index(id);
    if (f == previous_face)
      continue;
    previous_face = f;
    const auto &face = faces[f];
    for (Index i = 0, n = Index(face.size()); i < n; ++i) {
      if (Index(face[i]) != v)
        continue;
      ++corners;
      if (seed_face < 0) {
        seed_face = f;
        seed_corner = i;
      }
    }
  }
  if (corners == 0)
    return true;

  Index reached = 1;
  for (int side = 0; side < 2; ++side) {
    auto at = tf::topology::vertex_fan_position<Index>{seed_face, seed_corner,
                                                       side == 0};
    while (true) {
      Index peers = tf::topology::vertex_fan_step(faces, fm, v, at);
      if (peers > 1)
        return false;
      if (peers == 0)
        break;
      if (at.face == seed_face && at.corner == seed_corner)
        return reached == corners;
      // Past the corner count the walk has re-entered a corner, so it is
      // circling something that is not this vertex's one fan.
      if (++reached > corners)
        return false;
    }
  }
  return reached == corners;
}

} // namespace tf::topology
