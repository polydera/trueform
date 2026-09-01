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

/// The source faces a retired original vertex reaches and the world of the
/// asking tier never named.
///
/// A weld moves an original vertex onto another identity. Every source face
/// incident to that vertex holds an instance of it, so the source mesh's own
/// vertex -> faces membership IS the ring the identity leaves the cut world
/// to reach, and the mask alone decides which of them is new: a face the
/// world named already states the vertex through its own definitions.
template <typename Index, typename Retired, typename VertexOffsets,
          typename FaceOffsets, typename ApplyToForm, typename IsNamed>
auto discover_weld_entrants(const Retired &retired_originals,
                            const VertexOffsets &vertex_offsets,
                            const FaceOffsets &face_offsets,
                            const ApplyToForm &apply_to_form,
                            const IsNamed &is_named,
                            tf::buffer<Index> &entrants) -> void {
  if (retired_originals.size() == 0)
    return;
  tf::generic_generate(
      retired_originals, entrants,
      [&](Index flat, tf::buffer<Index> &out) {
        const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
        const auto vertex = flat - vertex_offsets[std::size_t(tag)];
        apply_to_form(tag, [&](const auto &form) {
          for (const auto face : form.face_membership()[vertex])
            if (!is_named(Index(tag), Index(face)))
              out.push_back(face_offsets[std::size_t(tag)] + Index(face));
        });
      },
      tf::checked);
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
/// An instance names its face-local edge, so the manifold link answers by
/// that index alone. Only a non-manifold edge — no single peer — pays the
/// membership intersection.
template <typename Index, typename Graph, typename Roots, typename FaceOffsets,
          typename ApplyToForm, typename IsNamed>
auto discover_split_entrants(const Graph &g, const Roots &split_roots,
                             const FaceOffsets &face_offsets,
                             const ApplyToForm &apply_to_form,
                             const IsNamed &is_named,
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
            if (descriptor.tag == tag && descriptor.object == object)
              return true;
          }
          return false;
        };
        for (const auto &instance : span) {
          if (instance.side < 0)
            continue;
          const auto &descriptor = descriptors[std::size_t(instance.face)];
          const auto tag = descriptor.tag;
          const auto face = descriptor.object;
          apply_to_form(tag, [&](const auto &form) {
            const auto corners = form.faces()[face];
            const auto n = corners.size();
            if (std::size_t(instance.side) >= n)
              return;
            const auto peer =
                form.manifold_edge_link()[face][std::size_t(instance.side)];
            if (peer.is_boundary())
              return;
            if (peer.is_simple()) {
              if (!in_group(tag, peer.face_peer) &&
                  !is_named(tag, peer.face_peer))
                out.push_back(face_offsets[std::size_t(tag)] + peer.face_peer);
              return;
            }
            const auto v0 = Index(corners[std::size_t(instance.side)]);
            const auto v1 =
                Index(corners[(std::size_t(instance.side) + 1) % n]);
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
          typename IsNamed>
auto discover_uncut_entrants(const Graph &g, const Retired &retired_originals,
                             const Roots &split_roots,
                             const VertexOffsets &vertex_offsets,
                             const FaceOffsets &face_offsets,
                             const ApplyToForm &apply_to_form,
                             const IsNamed &is_named,
                             tf::buffer<Index> &entrants) -> void {
  discover_weld_entrants(retired_originals, vertex_offsets, face_offsets,
                         apply_to_form, is_named, entrants);
  discover_split_entrants(g, split_roots, face_offsets, apply_to_form, is_named,
                          entrants);
  if (entrants.size() == 0)
    return;
  tbb::parallel_sort(entrants.begin(), entrants.end());
  entrants.erase_till_end(std::unique(entrants.begin(), entrants.end()));
}

} // namespace tf::intersect::graph
