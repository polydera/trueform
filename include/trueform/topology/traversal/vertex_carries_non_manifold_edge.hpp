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
#include "../../core/algorithm/circular_decrement.hpp"
#include "../../core/algorithm/circular_increment.hpp"
#include "../face_edge_neighbors.hpp"
#include "../face_membership_like.hpp"
#include <array>

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief Whether an edge at a vertex is shared by three or more faces.
///
/// @tparam Index The integer type for indices.
/// @tparam Faces The faces range type.
/// @tparam Policy The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @param v The vertex.
/// @return `true` if some edge at `v` carries three or more faces.
template <typename Index, typename Faces, typename Policy>
auto vertex_carries_non_manifold_edge(const Faces &faces,
                                      const tf::face_membership_like<Policy> &fm,
                                      const Index &v) -> bool {
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
      for (Index w : std::array<Index, 2>{
               Index(face[tf::circular_decrement(i, n)]),
               Index(face[tf::circular_increment(i, n)])}) {
        Index peers = 0;
        tf::face_edge_neighbors_apply(fm, faces, f, v, w,
                                      [&peers](const auto &) {
                                        return ++peers > 1;
                                      });
        if (peers > 1)
          return true;
      }
    }
  }
  return false;
}

} // namespace tf::topology
