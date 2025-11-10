/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/buffer.hpp"
#include "../core/faces.hpp"
#include "./directed_edge_id_in_face.hpp"
#include "./manifold_edge_link.hpp"
namespace tf {

template <typename Policy, typename Policy1>
void orient_faces_consistently(
    tf::faces<Policy> &faces,
    const tf::manifold_edge_link_like<Policy1> &link) {
  using Index = std::decay_t<decltype(link[0][0].face_peer)>;
  tf::buffer<bool> visited;
  visited.allocate(faces.size());
  tf::parallel_fill(visited, false);

  tf::buffer<Index> stack;
  stack.reserve(64);

  for (Index seed = 0; seed < static_cast<Index>(faces.size()); ++seed) {
    if (visited[seed])
      continue;

    visited[seed] = true;
    stack.push_back(seed);

    while (!stack.empty()) {
      Index curr = stack.back();
      stack.pop_back();

      const auto &curr_face = faces[curr];
      const auto &curr_link = link[curr];

      Index size = link.size();
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
        Index dir = tf::directed_edge_id_in_face(neighbor_face, b, a);
        if (dir == Index(neighbor_face.size())) {
          // Edge direction reversed → flip neighbor face
          std::reverse(neighbor_face.begin(), neighbor_face.end());
        }
        visited[neighbor] = true;
        stack.push_back(neighbor);
      }
    }
  }
}

template <typename Policy, typename Index, int N>
void orient_faces_consistently(tf::faces<Policy> &&faces,
                               const tf::manifold_edge_link<Index, N> &link) {
  return orient_faces_consistently(faces, link);
}
} // namespace tf
