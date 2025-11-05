/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/generic_generate.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/faces.hpp"
#include "../core/small_vector.hpp"
#include "../core/views/enumerate.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership.hpp"
#include "./policy/face_membership.hpp"

namespace tf {
template <typename Policy, typename Index>
auto make_non_manifold_edges(const tf::faces<Policy> &faces,
                             const tf::face_membership<Index> &fm) {
  tf::blocked_buffer<Index, 2> edges;
  tf::generic_generate(
      tf::enumerate(faces), edges.data_buffer(), tf::small_vector<Index, 10>{},
      [&](const auto &pair, auto &buffer, auto &neighbors) {
        const auto &[face_id, face] = pair;
        Index size = face.size();
        Index prev = size - 1;
        for (Index i = 0; i < size; prev = i++) {
          neighbors.clear();
          tf::face_edge_neighbors(fm, faces, Index(face_id), Index(face[prev]),
                                  Index(face[i]),
                                  std::back_inserter(neighbors));
          if (neighbors.size() > 1 &&
              // only keep a single copy of an edge
              std::all_of(neighbors.begin(), neighbors.end(),
                          [face_id = Index(face_id)](const auto &x) {
                            return x > face_id;
                          })) {
            buffer.push_back(face[prev]);
            buffer.push_back(face[i]);
          }
        }
      });
  return edges;
}

template <typename Policy>
auto make_non_manifold_edges(const tf::polygons<Policy> &polygons) {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::make_non_manifold_edges(polygons.faces(),
                                       polygons.face_membership());
  } else {
    tf::face_membership<std::decay_t<decltype(polygons.faces()[0][0])>> fe;
    fe.build(polygons);
    return tf::make_non_manifold_edges(polygons.faces(), fe);
  }
}
} // namespace tf
