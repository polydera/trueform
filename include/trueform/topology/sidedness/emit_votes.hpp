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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/tagged_sidedness.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/classify_wedge.hpp"
#include <tuple>

namespace tf::topology::sidedness {

/// @ingroup topology_components
/// @brief Walk non-manifold edges; emit per-edge sidedness votes.
///
/// At each non-manifold edge, every same-tag pair of incident faces
/// whose loops traverse the edge in opposite directions defines one
/// wedge — the v0→v1 face supplies the `wa` anchor, the v1→v0 face
/// supplies `wb`. For each such wedge, every face from a different
/// tag in the same fan classifies its off-edge vertex against the
/// wedge via `tf::exact::classify_wedge` and emits one vote:
///
///   vote_components.push_back(component of the classified face)
///   vote_entries.push_back({ wedge tag, sidedness })
///
/// Anchor assignment is canonical (loop direction picks `wa` vs.
/// `wb`), so the predicate result is deterministic regardless of the
/// order in which faces appear in the fan.
///
/// Multiple opposite-direction pairs at a single edge (k > 2 same-
/// tag faces, only possible if a single mesh is itself non-manifold
/// at this edge) produce multiple votes that funnel into the same
/// `(component, tag)` bucket — the downstream majority resolves any
/// disagreement.
///
/// @tparam Policy Polygons policy (must expose triangular faces).
/// @tparam Range Tag-labels range (per arrangement face).
/// @tparam Labels connected_component_labels range over arr faces.
/// @tparam EdgesRange Non-manifold edges (`blocked_buffer<Index,2>`).
/// @tparam FacesRange Non-manifold edge fans
///         (`offset_block_buffer<Index, Index>`).
/// @tparam GetPoint Functor `Index -> tf::point<ExactInt, 3>`.
/// @param polygons The arrangement.
/// @param tag_labels Per-face operand tag.
/// @param component_labels Per-face component id.
/// @param nm_edges Endpoint pairs per non-manifold edge.
/// @param nm_edge_faces Incident face fans per non-manifold edge.
/// @param get_point Vertex-id → exact point.
/// @param vote_components Out: my_component per emitted vote.
/// @param vote_entries Out: {your_tag, sidedness} per emitted vote.
template <typename Policy, typename Range, typename Labels, typename EdgesRange,
          typename FacesRange, typename GetPoint, typename Index>
auto emit_votes(const tf::polygons<Policy> &polygons, const Range &tag_labels,
                const Labels &component_labels, const EdgesRange &nm_edges,
                const FacesRange &nm_edge_faces, const GetPoint &get_point,
                tf::buffer<Index> &vote_components,
                tf::buffer<tf::tagged_sidedness<Index>> &vote_entries) -> void {
  tf::generic_generate(
      tf::zip(nm_edges, nm_edge_faces),
      std::tie(vote_components, vote_entries),
      [&](const auto &pair, auto &buffers) {
        auto &[components_buf, entries_buf] = buffers;
        auto &&[edge, face_block] = pair;

        Index v0_id = edge[0];
        Index v1_id = edge[1];
        auto v0 = get_point(v0_id);
        auto v1 = get_point(v1_id);

        auto off_vertex_id = [&](Index face_id) -> Index {
          auto face = polygons.faces()[face_id];
          for (auto v : face)
            if (v != v0_id && v != v1_id)
              return v;
          return Index(-1);
        };

        // Direction the face's loop traverses the NM edge:
        //   0 → loop has v0_id → v1_id  (wa side)
        //   1 → loop has v1_id → v0_id  (wb side)
        //  -1 → edge not in face (shouldn't happen)
        auto edge_dir_in_face = [&](Index face_id) -> int {
          auto face = polygons.faces()[face_id];
          auto n_face = face.size();
          for (std::size_t i = 0; i < n_face; ++i) {
            if (face[i] == v0_id)
              return face[(i + 1) % n_face] == v1_id ? 0 : 1;
          }
          return -1;
        };

        auto n = face_block.size();
        for (std::size_t a = 0; a < n; ++a) {
          if (edge_dir_in_face(face_block[a]) != 0)
            continue;
          auto wedge_tag = tag_labels[face_block[a]];
          auto wa = get_point(off_vertex_id(face_block[a]));
          for (std::size_t b = 0; b < n; ++b) {
            if (b == a || tag_labels[face_block[b]] != wedge_tag ||
                edge_dir_in_face(face_block[b]) != 1)
              continue;
            auto wb = get_point(off_vertex_id(face_block[b]));
            for (std::size_t k = 0; k < n; ++k) {
              if (tag_labels[face_block[k]] == wedge_tag)
                continue;
              auto F_id = face_block[k];
              auto F_off = get_point(off_vertex_id(F_id));
              auto side = tf::exact::classify_wedge(v0, v1, wa, wb, F_off);
              components_buf.push_back(component_labels.labels[F_id]);
              entries_buf.push_back({wedge_tag, side});
            }
          }
        }
      });
}

} // namespace tf::topology::sidedness
