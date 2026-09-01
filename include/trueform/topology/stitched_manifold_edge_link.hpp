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
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/direction.hpp"
#include "../core/faces.hpp"
#include "../core/small_vector.hpp"
#include "../core/static_size.hpp"
#include "../core/stitch_index_map.hpp"
#include "../core/views/drop.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/slide_range.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership_like.hpp"
#include "./manifold_edge_link.hpp"
#include "./manifold_edge_link_like.hpp"
#include "./manifold_edge_peer.hpp"
#include <algorithm>
#include <cstddef>

namespace tf {

/// @ingroup topology_connectivity
/// @brief Manifold edge link of a merged mesh, reusing the sources'.
///
/// A face that survived whole keeps its peers, remapped through `sim.face_f`.
/// A peer is only reusable when it survived too and no endpoint of the shared
/// edge was touched by a new face; everything else is recomputed from the
/// stitched membership.
///
/// Unlike membership and spatial trees, this structure is indexed by position
/// *within* a face: peer slot `i` belongs to the edge leaving vertex `i`. So it
/// is the one consumer that has to know whether an output face kept its source
/// winding — hence `direction0` / `direction1`, which after a boolean come from
/// @ref tf::make_directions.
///
/// @tparam Index The integer type for indices.
/// @param result_faces The faces of the merged mesh.
/// @param mel0 Manifold edge link of the first source mesh.
/// @param mel1 Manifold edge link of the second source mesh.
/// @param fm_stitched The merged mesh's @ref tf::stitched_face_membership.
/// @param sim The merge's @ref tf::stitch_index_map (two sources).
/// @param direction0 Winding of the first source's surviving faces.
/// @param direction1 Winding of the second source's surviving faces.
template <typename Index, typename FacesPolicy, typename MELPolicy0,
          typename MELPolicy1, typename FMPolicy>
auto stitched_manifold_edge_link(
    const tf::faces<FacesPolicy> &result_faces,
    const tf::manifold_edge_link_like<MELPolicy0> &mel0,
    const tf::manifold_edge_link_like<MELPolicy1> &mel1,
    const tf::face_membership_like<FMPolicy> &fm_stitched,
    const tf::stitch_index_map<Index> &sim, tf::direction direction0,
    tf::direction direction1) {
  constexpr std::size_t N0 = tf::static_size_v<decltype(mel0[0])>;
  constexpr std::size_t N1 = tf::static_size_v<decltype(mel1[0])>;
  constexpr std::size_t NResult =
      (N0 == tf::dynamic_size || N1 == tf::dynamic_size || N0 != N1)
          ? tf::dynamic_size
          : N0;

  const Index n_faces = sim.n_output_faces;
  const Index gone = n_faces;
  // Not constexpr: a local constexpr odr-used by a lambda is MSVC C3493.
  const Index needs_recompute = Index(-4);

  auto with_mel = [&](Index source, auto &&fn) {
    if (source == 0)
      fn(mel0);
    else
      fn(mel1);
  };
  auto is_reversed = [&](Index source) {
    return (source == 0 ? direction0 : direction1) == tf::direction::reverse;
  };

  // A new face's vertices carry no reusable incidence, so every edge touching
  // one has to be recomputed.
  tf::buffer<bool> new_point_mask;
  new_point_mask.allocate(fm_stitched.size());
  tf::parallel_fill(new_point_mask, false);
  tf::parallel_for_each(
      sim.new_faces,
      [&](Index f) {
        for (auto v : result_faces[f])
          new_point_mask[v] = true;
      },
      tf::checked);

  tf::manifold_edge_link<Index, NResult> result;
  if constexpr (NResult == tf::dynamic_size) {
    result.offsets_buffer().allocate(result_faces.size() + 1);
    result.offsets_buffer()[0] = 0;
    tf::parallel_transform(
        result_faces, tf::drop(result.offsets_buffer(), 1),
        [](const auto &face) { return Index(face.size()); }, tf::checked);
    for (auto &&[a, b] : tf::make_slide_range<2>(result.offsets_buffer()))
      b += a;
    result.data_buffer().allocate(result.offsets_buffer().back());
  } else {
    result.data_buffer().allocate(result_faces.size() * NResult);
  }

  auto touches_new = [&](Index v0, Index v1) {
    return new_point_mask[v0] || new_point_mask[v1];
  };

  // A reversed face lists [v0,v1,...,v_{n-1}] as [v_{n-1},...,v1,v0], which
  // sends edge slot i to n-2-i, except the closing slot n-1, which is its own.
  auto reversed_edge_slot = [](Index i, Index n) -> Index {
    return (i < n - 1) ? (n - 2 - i) : (n - 1);
  };

  for (Index s = 0; s < sim.n_sources; ++s) {
    const bool reversed = is_reversed(s);
    auto forward = sim.face_f[s];
    with_mel(s, [&](const auto &mel) {
      tf::parallel_for_each(
          tf::make_sequence_range(Index(0), Index(forward.size())),
          [&](Index source_face) {
            const Index f = forward[source_face];
            if (f == gone)
              return;
            const auto &source_peers = mel[source_face];
            auto &&peers = result[f];
            const auto &face = result_faces[f];
            const Index n_edges = Index(face.size());

            Index current = n_edges - 1;
            for (Index next = 0; next < n_edges; current = next++) {
              const Index source_slot =
                  reversed ? reversed_edge_slot(current, n_edges) : current;
              const Index source_peer = source_peers[source_slot].face_peer;

              if (touches_new(face[current], face[next]) ||
                  source_peer == tf::manifold_edge_peer<Index>::non_manifold ||
                  source_peer == tf::manifold_edge_peer<
                                     Index>::non_manifold_representative) {
                peers[current] = {needs_recompute};
              } else if (source_peer ==
                         tf::manifold_edge_peer<Index>::boundary) {
                peers[current] = {tf::manifold_edge_peer<Index>::boundary};
              } else if (forward[source_peer] == gone) {
                peers[current] = {needs_recompute};
              } else {
                peers[current] = {forward[source_peer]};
              }
            }
          },
          tf::checked);
    });
  }

  tf::parallel_for_each(
      sim.new_faces,
      [&](Index f) {
        auto &&peers = result[f];
        const Index n_edges = Index(result_faces[f].size());
        for (Index edge = 0; edge < n_edges; ++edge)
          peers[edge] = {needs_recompute};
      },
      tf::checked);

  tf::parallel_for(
      tf::make_sequence_range(Index(0), n_faces), [&](auto begin, auto end) {
        tf::small_vector<Index, 6> neighbors;
        while (begin != end) {
          Index face_id = *begin++;
          auto &&peers = result[face_id];
          const auto &face = result_faces[face_id];
          const Index n_edges = Index(face.size());

          Index current = n_edges - 1;
          for (Index next = 0; next < n_edges; current = next++) {
            if (peers[current].face_peer != needs_recompute)
              continue;

            neighbors.clear();
            tf::face_edge_neighbors(fm_stitched, result_faces, face_id,
                                    Index(face[current]), Index(face[next]),
                                    std::back_inserter(neighbors));

            switch (neighbors.size()) {
            case 0:
              peers[current] = {tf::manifold_edge_peer<Index>::boundary};
              break;
            case 1:
              peers[current] = {neighbors[0]};
              break;
            default:
              if (std::all_of(neighbors.begin(), neighbors.end(),
                              [&](const auto &x) { return x > face_id; }))
                peers[current] = {
                    tf::manifold_edge_peer<Index>::non_manifold_representative};
              else
                peers[current] = {tf::manifold_edge_peer<Index>::non_manifold};
              break;
            }
          }
        }
      });

  return result;
}

} // namespace tf
