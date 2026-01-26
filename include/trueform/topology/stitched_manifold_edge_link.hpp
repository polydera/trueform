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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/buffer.hpp"
#include "../core/faces.hpp"
#include "../core/small_vector.hpp"
#include "../core/static_size.hpp"
#include "../core/stitch_index_maps.hpp"
#include "../core/views/drop.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/slide_range.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership_like.hpp"
#include "./manifold_edge_link.hpp"
#include "./manifold_edge_link_like.hpp"
#include "./manifold_edge_peer.hpp"
#include <algorithm>

namespace tf {

/// @ingroup topology_connectivity
/// @brief Stitch manifold edge link from two source meshes using stitch index maps.
///
/// Combines manifold edge link structures from two meshes being stitched together.
/// Uses the stitch index maps to remap face indices, and recomputes edge
/// connectivity for edges affected by the stitching operation ("dirty" edges
/// that touch vertices near the stitch boundary).
///
/// @tparam Index The integer type for indices.
/// @tparam FacesPolicy The result faces policy type.
/// @tparam MELPolicy0 The first manifold edge link policy type.
/// @tparam MELPolicy1 The second manifold edge link policy type.
/// @tparam FMPolicy The stitched face membership policy type.
/// @param result_faces The faces of the stitched result mesh.
/// @param mel0 Manifold edge link from the first source mesh.
/// @param mel1 Manifold edge link from the second source mesh.
/// @param fm_stitched The stitched face membership structure.
/// @param im The stitch index maps for remapping.
/// @return A new manifold edge link structure for the stitched mesh.
template <typename Index, typename FacesPolicy, typename MELPolicy0,
          typename MELPolicy1, typename FMPolicy>
auto stitched_manifold_edge_link(
    const tf::faces<FacesPolicy> &result_faces,
    const tf::manifold_edge_link_like<MELPolicy0> &mel0,
    const tf::manifold_edge_link_like<MELPolicy1> &mel1,
    const tf::face_membership_like<FMPolicy> &fm_stitched,
    const tf::stitch_index_maps<Index> &im) {
  constexpr std::size_t N0 = tf::static_size_v<decltype(mel0[0])>;
  constexpr std::size_t N1 = tf::static_size_v<decltype(mel1[0])>;
  constexpr std::size_t NResult =
      (N0 == tf::dynamic_size || N1 == tf::dynamic_size || N0 != N1)
          ? tf::dynamic_size
          : N0;

  const Index dirty_start =
      im.polygons0.kept_ids().size() + im.polygons1.kept_ids().size();
  const Index n_dirty = result_faces.size() - dirty_start;

  // Build dirty point mask
  tf::buffer<bool> dirty_point_mask;
  dirty_point_mask.allocate(fm_stitched.size());
  tf::parallel_fill(dirty_point_mask, false);

  tf::parallel_for_each(
      tf::make_sequence_range(n_dirty),
      [&](Index i) {
        const auto &face = result_faces[dirty_start + i];
        for (auto v : face)
          dirty_point_mask[v] = true;
      },
      tf::checked);

  // Allocate result
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

  constexpr Index sentinel = Index(-1);
  constexpr Index needs_recompute = Index(-4);

  auto is_dirty_edge = [&](Index v0, Index v1) {
    return dirty_point_mask[v0] || dirty_point_mask[v1];
  };

  const bool mesh0_flipped = im.direction0 == tf::direction::reverse;
  const bool mesh1_flipped = im.direction1 == tf::direction::reverse;

  // When a face is reversed, edge indices are remapped.
  // For [v0,v1,...,v_{n-1}] → [v_{n-1},...,v1,v0]:
  // orig_edge = (n-2-i) for i<n-1, else n-1
  auto flipped_edge_index = [](Index i, Index n) -> Index {
    return (i < n - 1) ? (n - 2 - i) : (n - 1);
  };

  // Copy & remap clean faces from mesh0
  tf::parallel_for_each(
      im.polygons0.kept_ids(),
      [&](Index orig_face_id) {
        Index result_face_id =
            im.polygons0.f()[orig_face_id] + im.polygons0_offset;
        const auto &orig_peers = mel0[orig_face_id];
        auto &&result_peers = result[result_face_id];
        const auto &face = result_faces[result_face_id];
        const Index n_edges = face.size();

        Index current = n_edges - 1;
        for (Index next = 0; next < n_edges; current = next++) {
          Index orig_edge =
              mesh0_flipped ? flipped_edge_index(current, n_edges) : current;
          Index orig_peer = orig_peers[orig_edge].face_peer;

          if (is_dirty_edge(face[current], face[next]) ||
              orig_peer == tf::manifold_edge_peer<Index>::non_manifold ||
              orig_peer ==
                  tf::manifold_edge_peer<Index>::non_manifold_representative) {
            result_peers[current] = {needs_recompute};
          } else if (orig_peer == tf::manifold_edge_peer<Index>::boundary) {
            result_peers[current] = {tf::manifold_edge_peer<Index>::boundary};
          } else {
            Index remapped_peer = im.polygons0.f()[orig_peer];
            if (remapped_peer == sentinel) {
              result_peers[current] = {needs_recompute};
            } else {
              result_peers[current] = {remapped_peer + im.polygons0_offset};
            }
          }
        }
      },
      tf::checked);

  // Copy & remap clean faces from mesh1
  tf::parallel_for_each(
      im.polygons1.kept_ids(),
      [&](Index orig_face_id) {
        Index result_face_id =
            im.polygons1.f()[orig_face_id] + im.polygons1_offset;
        const auto &orig_peers = mel1[orig_face_id];
        auto &&result_peers = result[result_face_id];
        const auto &face = result_faces[result_face_id];
        const Index n_edges = face.size();

        Index current = n_edges - 1;
        for (Index next = 0; next < n_edges; current = next++) {
          Index orig_edge =
              mesh1_flipped ? flipped_edge_index(current, n_edges) : current;
          Index orig_peer = orig_peers[orig_edge].face_peer;

          if (is_dirty_edge(face[current], face[next]) ||
              orig_peer == tf::manifold_edge_peer<Index>::non_manifold ||
              orig_peer ==
                  tf::manifold_edge_peer<Index>::non_manifold_representative) {
            result_peers[current] = {needs_recompute};
          } else if (orig_peer == tf::manifold_edge_peer<Index>::boundary) {
            result_peers[current] = {tf::manifold_edge_peer<Index>::boundary};
          } else {
            Index remapped_peer = im.polygons1.f()[orig_peer];
            if (remapped_peer == sentinel) {
              result_peers[current] = {needs_recompute};
            } else {
              result_peers[current] = {remapped_peer + im.polygons1_offset};
            }
          }
        }
      },
      tf::checked);

  // Initialize dirty faces as needing recomputation
  tf::parallel_for_each(
      tf::make_sequence_range(n_dirty),
      [&](Index i) {
        Index result_face_id = dirty_start + i;
        auto &&result_peers = result[result_face_id];
        const Index n_edges = result_faces[result_face_id].size();
        for (Index edge = 0; edge < n_edges; ++edge) {
          result_peers[edge] = {needs_recompute};
        }
      },
      tf::checked);

  // Recompute edges marked as needs_recompute
  tf::parallel_for(
      tf::make_sequence_range(result_faces.size()), [&](auto begin, auto end) {
        tf::small_vector<Index, 6> neighbors;
        while (begin != end) {
          Index face_id = *begin++;
          auto &&peers = result[face_id];
          const auto &face = result_faces[face_id];
          const Index n_edges = face.size();

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
