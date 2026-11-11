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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../topology/face_edge_neighbors.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::intersect::graph {

/// THE MASK, LA-side: whether the cut world NAMED a source face. Its
/// descriptors ascend by `(tag, object)` — @ref plane_graph builds them in
/// the face carrier's own order — so the flat key ascends with them and one
/// binary search answers, with nothing materialized.
///
/// The producers below ask a mask as they emit, because NOT IN THE GROUP IS
/// NOT THE SAME AS NOT IN THE WORLD: each decides who is new against its own
/// facts — a split against its group's instances, a weld against the source
/// mesh's own ring — and a face can hold that edge or that vertex, own a
/// definition somewhere, and appear in neither. Promoting one the world
/// already names would state a single `(tag, object)` twice, which the
/// exposure reads as one slot.
template <typename Index, typename Descriptors, typename FaceOffsets>
auto plane_graph_names_face(const Descriptors &descriptors,
                            const FaceOffsets &face_offsets, Index tag,
                            Index object) -> bool {
  const auto key = face_offsets[std::size_t(tag)] + object;
  const auto at = std::lower_bound(
      descriptors.begin(), descriptors.end(), key,
      [&face_offsets](const auto &descriptor, Index k) {
        return face_offsets[std::size_t(descriptor.tag)] + descriptor.object <
               k;
      });
  return at != descriptors.end() &&
         face_offsets[std::size_t(at->tag)] + at->object == key;
}

/// The faces a retired original vertex reaches and the world of the asking
/// tier never named.
///
/// A weld moves an original vertex onto another identity. Every face
/// incident to that vertex holds an instance of it, so the vertex's faces
/// ARE the ring the identity leaves the cut world to reach, and the mask
/// alone decides which of them is new: a face the world named already states
/// the vertex through its own definitions. `holders(vertex, f)` calls
/// `f(face, entrant)` for every face holding `vertex`, `face` as `is_named`
/// reads it and `entrant` the key appended to `entrants`, unordered and
/// unique only per vertex.
template <typename Index, typename Retired, typename Holders, typename IsNamed>
auto discover_weld_entrants(const Retired &retired_originals,
                            const Holders &holders, const IsNamed &is_named,
                            tf::buffer<Index> &entrants) -> void {
  if (retired_originals.size() == 0)
    return;
  tf::generic_generate(
      retired_originals, entrants,
      [&](Index flat, tf::buffer<Index> &out) {
        holders(flat, [&](const auto &face, Index entrant) {
          if (!is_named(face))
            out.push_back(entrant);
        });
      },
      tf::checked);
}

/// The source faces a retired original vertex reaches, read through the
/// source mesh's own vertex -> faces membership: per flat vertex its form's
/// `face_membership()` row, each face keyed by its form's face offset.
template <typename Index, typename Retired, typename VertexOffsets,
          typename FaceOffsets, typename ApplyToForm, typename IsNamed>
auto discover_weld_entrants(const Retired &retired_originals,
                            const VertexOffsets &vertex_offsets,
                            const FaceOffsets &face_offsets,
                            const ApplyToForm &apply_to_form,
                            const IsNamed &is_named,
                            tf::buffer<Index> &entrants) -> void {
  discover_weld_entrants<Index>(
      retired_originals,
      [&](Index flat, const auto &f) {
        const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
        const auto own = flat - vertex_offsets[std::size_t(tag)];
        apply_to_form(tag, [&](const auto &form) {
          for (const auto face : form.face_membership()[own])
            f(std::array<Index, 2>{Index(tag), Index(face)},
              face_offsets[std::size_t(tag)] + Index(face));
        });
      },
      [&is_named](const std::array<Index, 2> &face) {
        return is_named(face[0], face[1]);
      },
      entrants);
}

/// The source faces a split on an original edge reaches and the world of the
/// asking tier never named.
///
/// A cut root that lies on an original edge states a new identity on that
/// edge, and an edge belongs to every face that holds it. The group's own
/// instances already carry it, so the faces on the edge that hold no
/// instance of the group are the ones the split leaves the cut world to
/// reach.
///
/// An instance names its face-local edge, read on the source face
/// `source_side(face, side)` states, so the manifold link answers by that
/// index alone. Only a non-manifold edge — no single peer — pays the
/// membership intersection.
template <typename Index, typename Graph, typename Roots, typename FaceOffsets,
          typename ApplyToForm, typename IsNamed, typename SourceSide>
auto discover_split_entrants(const Graph &g, const Roots &split_roots,
                             const FaceOffsets &face_offsets,
                             const ApplyToForm &apply_to_form,
                             const IsNamed &is_named,
                             const SourceSide &source_side,
                             tf::buffer<Index> &entrants) -> void {
  if (split_roots.size() == 0)
    return;
  const auto descriptors = g.descriptors();
  tf::generic_generate(
      split_roots, entrants,
      [&](Index root, tf::buffer<Index> &out) {
        const auto span = g.canon_group(root);
        auto in_group = [&](Index tag, Index object) {
          for (const auto &instance : span) {
            const auto &descriptor = descriptors[std::size_t(instance.face)];
            if (descriptor.tag == tag &&
                source_side(Index(instance.face), Index(0))[0] == object)
              return true;
          }
          return false;
        };
        for (const auto &instance : span) {
          if (instance.side < 0)
            continue;
          // a side a carve split within its source side shares that edge
          // only with faces the entrance already brought in
          const auto source =
              source_side(Index(instance.face), Index(instance.side));
          if (source[1] < Index(0))
            continue;
          const auto &descriptor = descriptors[std::size_t(instance.face)];
          const auto tag = descriptor.tag;
          const auto face = source[0];
          apply_to_form(tag, [&](const auto &form) {
            const auto corners = form.faces()[face];
            const auto n = corners.size();
            const auto side = std::size_t(source[1]);
            if (side >= n)
              return;
            const auto peer = form.manifold_edge_link()[face][side];
            if (peer.is_boundary())
              return;
            if (peer.is_simple()) {
              if (!in_group(tag, peer.face_peer) &&
                  !is_named(tag, peer.face_peer))
                out.push_back(face_offsets[std::size_t(tag)] + peer.face_peer);
              return;
            }
            const auto v0 = Index(corners[side]);
            const auto v1 = Index(corners[(side + 1) % n]);
            tf::face_edge_neighbors_apply(
                form.face_membership(), form.faces(), face, v0, v1,
                [&](const auto &neighbour) {
                  if (!in_group(tag, Index(neighbour)) &&
                      !is_named(tag, Index(neighbour)))
                    out.push_back(face_offsets[std::size_t(tag)] +
                                  Index(neighbour));
                  return false;
                });
          });
        }
      },
      tf::checked);
}

/// THE ENTRANCE DISCOVERY, one for both tiers: every source face the asking
/// world never named that a retired original or a split on an original edge
/// nevertheless reaches, each stated once.
///
/// `is_named` is the asking tier's own authority on who its world holds —
/// @ref plane_graph_names_face for the cut world's descriptors, the dense
/// answered mask for a plane wave — and nothing else filters. Both producers
/// emit into one stream — a face bordering two split edges, or found through
/// several instances of one, is discovered as many times as it is reached —
/// and the sort states it once.
template <typename Index, typename Graph, typename Retired, typename Roots,
          typename VertexOffsets, typename FaceOffsets, typename ApplyToForm,
          typename IsNamed, typename SourceSide>
auto discover_uncut_entrants(const Graph &g, const Retired &retired_originals,
                             const Roots &split_roots,
                             const VertexOffsets &vertex_offsets,
                             const FaceOffsets &face_offsets,
                             const ApplyToForm &apply_to_form,
                             const IsNamed &is_named,
                             const SourceSide &source_side,
                             tf::buffer<Index> &entrants) -> void {
  discover_weld_entrants(retired_originals, vertex_offsets, face_offsets,
                         apply_to_form, is_named, entrants);
  discover_split_entrants(g, split_roots, face_offsets, apply_to_form, is_named,
                          source_side, entrants);
  if (entrants.size() == 0)
    return;
  tbb::parallel_sort(entrants.begin(), entrants.end());
  entrants.erase_till_end(std::unique(entrants.begin(), entrants.end()));
}

/// The carved faces the cut world never named, joined to the entrants:
/// a re-seat's pieces enter whether a record reaches them or not.
/// `carved_faces` and `entrants` are flat face keys, ascending.
template <typename Index, typename Descriptors, typename FaceOffsets,
          typename CarvedFaces>
auto discover_carved_entrants(const Descriptors &descriptors,
                              const FaceOffsets &face_offsets,
                              const CarvedFaces &carved_faces,
                              tf::buffer<Index> &entrants) -> void {
  if (carved_faces.size() == 0)
    return;
  for (const auto key : carved_faces) {
    const auto tag = tf::exact::tag_of_flat_vertex(face_offsets, Index(key));
    if (!plane_graph_names_face<Index>(descriptors, face_offsets, Index(tag),
                                       Index(key) -
                                           face_offsets[std::size_t(tag)]))
      entrants.push_back(Index(key));
  }
  tbb::parallel_sort(entrants.begin(), entrants.end());
  entrants.erase_till_end(std::unique(entrants.begin(), entrants.end()));
}

} // namespace tf::intersect::graph
