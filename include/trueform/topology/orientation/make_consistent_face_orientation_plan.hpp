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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/faces.hpp"
#include "../directed_edge_id_in_face.hpp"
#include "../manifold_edge_link_like.hpp"
#include <cstddef>
#include <type_traits>

namespace tf::topology {

/// @ingroup topology_analysis
/// @brief The reversal decided for every face, before any face is reversed.
struct consistent_face_orientation_plan {
  /// @brief Per-face reversal, aligned one-to-one with the planned faces.
  tf::buffer<bool> flip_mask;
  /// @brief False when any component's manifold-edge cycles contradict.
  bool orientable = true;
};

/// @ingroup topology_analysis
/// @brief Decide every face's reversal against the untouched winding.
///
/// The link's slot i names the edge leaving vertex i of the winding it was
/// built from, so the walk reads both and writes neither. Every parity is then
/// a function of the input alone, and each manifold-edge component takes the
/// side its weights vote for. A component whose parity cycles contradict has
/// no consistent winding to reach, so its mask stays empty and it reports the
/// verdict.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The manifold edge link policy type.
/// @tparam Range The weights range type.
/// @param faces The faces range.
/// @param link The manifold edge link built from exactly those face slots.
/// @param weights Per-face weights for voting (e.g., face areas).
/// @return The per-face reversal mask and the orientability verdict.
template <typename Policy, typename Policy1, typename Range>
auto make_consistent_face_orientation_plan(
    const tf::faces<Policy> &faces,
    const tf::manifold_edge_link_like<Policy1> &link, const Range &weights)
    -> consistent_face_orientation_plan {
  using Index = std::decay_t<decltype(link[0][0].face_peer)>;
  using Weight = std::decay_t<decltype(weights[0])>;

  consistent_face_orientation_plan plan;
  plan.flip_mask.allocate(faces.size());

  tf::buffer<bool> visited;
  visited.allocate(faces.size());
  tf::parallel_fill(visited, false);

  tf::buffer<Index> component;
  component.reserve(64);

  for (Index seed = 0; seed < static_cast<Index>(faces.size()); ++seed) {
    if (visited[seed])
      continue;

    visited[seed] = true;
    plan.flip_mask[seed] = false;
    component.clear();
    component.push_back(seed);

    Weight weight_flipped = 0;
    Weight weight_not_flipped = weights[seed]; // seed is not flipped
    bool component_orientable = true;

    std::size_t queue_begin = 0;

    while (queue_begin < component.size()) {
      Index curr = component[queue_begin++];

      const auto &curr_face = faces[curr];
      const auto &curr_link = link[curr];

      Index size = curr_link.size();
      Index prev = size - 1;
      for (Index i = 0; i < size; prev = i++) {
        const auto &peer = curr_link[prev];
        if (!peer.is_simple())
          continue;

        Index neighbor = peer.face_peer;
        const auto &neighbor_face = faces[neighbor];
        Index a = curr_face[prev];
        Index b = curr_face[i];

        bool edge_requires_flip =
            tf::directed_edge_id_in_face(b, a, neighbor_face) ==
            Index(neighbor_face.size());
        bool neighbor_flip = plan.flip_mask[curr] != edge_requires_flip;

        if (visited[neighbor]) {
          if (plan.flip_mask[neighbor] != neighbor_flip)
            component_orientable = false;
          continue;
        }

        visited[neighbor] = true;
        plan.flip_mask[neighbor] = neighbor_flip;
        if (neighbor_flip)
          weight_flipped += weights[neighbor];
        else
          weight_not_flipped += weights[neighbor];
        component.push_back(neighbor);
      }
    }

    if (!component_orientable) {
      plan.orientable = false;
      for (Index face_id : component)
        plan.flip_mask[face_id] = false;
    } else if (weight_flipped > weight_not_flipped) {
      for (Index face_id : component)
        plan.flip_mask[face_id] = !plan.flip_mask[face_id];
    }
  }

  return plan;
}
} // namespace tf::topology
