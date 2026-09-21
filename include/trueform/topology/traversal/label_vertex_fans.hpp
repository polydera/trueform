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
#include "../../core/buffer.hpp"
#include "../face_membership_like.hpp"
#include "./vertex_fan_step.hpp"

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief A corner of a face at a vertex, and the fan the corner walks in.
///
/// @tparam Index The integer type for indices.
template <typename Index> struct vertex_fan_corner {
  Index face;
  Index corner;
  Index fan;
};

/// @ingroup topology_analysis
/// @brief State every corner at a vertex and the fan it walks in.
///
/// A seed takes the smallest face no fan has claimed, both of its sides are
/// stepped, and the corners they reach are that fan's. So fan `0` holds the
/// smallest face at the vertex and the rest follow in the order their own
/// smallest face seeds them.
///
/// @tparam Index The integer type for indices.
/// @tparam Faces The faces range type.
/// @tparam Policy The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @param v The vertex.
/// @param corners Receives one entry per corner at `v`.
/// @return The number of fans, and zero when an edge at `v` carries three
///   faces: no fan crosses it, so the fans it holds apart are not separable.
template <typename Index, typename Faces, typename Policy>
auto label_vertex_fans(const Faces &faces,
                       const tf::face_membership_like<Policy> &fm,
                       const Index &v,
                       tf::buffer<vertex_fan_corner<Index>> &corners) -> Index {
  corners.clear();
  Index previous_face = -1;
  for (const auto &id : fm[v]) {
    auto f = Index(id);
    if (f == previous_face)
      continue;
    previous_face = f;
    const auto &face = faces[f];
    for (Index i = 0, n = Index(face.size()); i < n; ++i)
      if (Index(face[i]) == v)
        corners.push_back(vertex_fan_corner<Index>{f, i, -1});
  }

  Index fans = 0;
  while (true) {
    Index seed = -1;
    for (Index k = 0, n = Index(corners.size()); k < n; ++k)
      if (corners[k].fan < 0 &&
          (seed < 0 || corners[k].face < corners[seed].face))
        seed = k;
    if (seed < 0)
      return fans;

    Index seed_face = corners[seed].face;
    Index seed_corner = corners[seed].corner;
    corners[seed].fan = fans;
    bool closed = false;
    for (int side = 0; side < 2 && !closed; ++side) {
      auto at = tf::topology::vertex_fan_position<Index>{seed_face, seed_corner,
                                                        side == 0};
      while (true) {
        Index peers = tf::topology::vertex_fan_step(faces, fm, v, at);
        if (peers > 1)
          return 0;
        if (peers == 0)
          break;
        if (at.face == seed_face && at.corner == seed_corner) {
          closed = true;
          break;
        }
        for (auto &corner : corners)
          if (corner.face == at.face && corner.corner == at.corner) {
            // Past its own corners the walk is circling something that is
            // not this vertex's fans.
            if (corner.fan >= 0)
              return 0;
            corner.fan = fans;
            break;
          }
      }
    }
    ++fans;
  }
}

} // namespace tf::topology
