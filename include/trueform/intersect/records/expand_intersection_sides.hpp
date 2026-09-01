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

#include "../../core/algorithm/circular_increment.hpp"
#include "../../topology/edge_id_in_face.hpp"
#include "../../topology/face_edge_neighbors.hpp"
#include "../../topology/topo_id.hpp"
#include "../../topology/topo_type.hpp"
#include "../../topology/vertex_id_in_face.hpp"
#include "./tagged_intersection.hpp"

#include <cstddef>
#include <iterator>

namespace tf::intersect {

/// One face a contact's feature reaches, and the feature's slot there.
template <typename Index> struct feature_instance {
  Index object;
  Index target_id;
};

/// The faces one side of a contact reaches, each with the contact's slot
/// in it.
///
/// A contact is a fact about a FEATURE, and a feature belongs to every
/// face incident to it: a vertex reaches its whole face fan, an edge its
/// manifold peer (or, on a non-manifold edge, every face standing on it),
/// an interior point only its own face. `skip` drops one face from a
/// vertex fan — the self pair's other side, which states the contact
/// itself.
template <typename Index, typename Faces, typename FE, typename MEL,
          typename Neighbors, typename Out>
auto expand_intersection_feature(const Faces &faces, const FE &fe,
                                 const MEL &mel, Index object,
                                 tf::topo_id<Index> target, Index skip,
                                 Neighbors &neighbors, Out &out) -> void {
  if (target.label == tf::topo_type::edge) {
    out.push_back({object, target.id});
    const auto n = Index(faces[object].size());
    const Index e0 = faces[object][std::size_t(target.id)];
    const Index e1 =
        faces[object][std::size_t(tf::circular_increment<Index>(target.id, n))];
    auto &&link = mel[object][std::size_t(target.id)];
    if (link.is_simple()) {
      const Index peer = link.face_peer;
      out.push_back({peer, Index(tf::edge_id_in_face(e1, e0, faces[peer]))});
    } else if (!link.is_manifold()) {
      neighbors.clear();
      tf::face_edge_neighbors(fe, faces, object, e0, e1,
                              std::back_inserter(neighbors));
      for (auto n_face : neighbors)
        out.push_back(
            {n_face, Index(tf::edge_id_in_face(e1, e0, faces[n_face]))});
    }
  } else if (target.label == tf::topo_type::vertex) {
    const Index vid = faces[object][std::size_t(target.id)];
    for (auto face_id : fe[vid]) {
      if (Index(face_id) == skip)
        continue;
      out.push_back(
          {Index(face_id),
           tf::vertex_id_in_face<Index>(vid, faces[Index(face_id)])});
    }
  } else {
    out.push_back({object, target.id});
  }
}

/// Whether a fan of these two extents is a PRODUCT worth gating.
///
/// The pair currency's gate costs a read per pair and buys the pairs it
/// drops, so it pays only where the product exceeds the sum it would
/// otherwise be stated as. An edge against a face, two crossing edges, a
/// vertex against a face interior — the whole transversal family — states
/// no more pairs than features, so it fans whole and asks the gate
/// nothing.
constexpr auto extents_make_a_product(std::size_t own, std::size_t other)
    -> bool {
  return (own - 1) * (other - 1) > 1;
}

/// How many faces one side of a contact reaches, without materializing
/// them — an UPPER bound, since @ref extents_make_a_product only rises
/// with an extent and a mark too many costs a ticket, never a chord.
template <typename Index, typename Faces, typename FE, typename MEL>
auto intersection_feature_extent(const Faces &faces, const FE &fe,
                                 const MEL &mel, Index object,
                                 tf::topo_id<Index> target) -> std::size_t {
  if (target.label == tf::topo_type::edge) {
    auto &&link = mel[object][std::size_t(target.id)];
    if (link.is_simple())
      return 2;
    if (link.is_manifold())
      return 1;
  } else if (target.label != tf::topo_type::vertex) {
    return 1;
  }
  return std::size_t(fe[faces[object][std::size_t(target.id)]].size());
}

/// Whether a contact's fan is a product, read off the two extents alone.
template <typename Index, typename Faces0, typename FE0, typename MEL0,
          typename Faces1, typename FE1, typename MEL1>
auto intersection_sides_are_a_product(const Faces0 &faces0, const FE0 &fe0,
                                      const MEL0 &mel0, const Faces1 &faces1,
                                      const FE1 &fe1, const MEL1 &mel1,
                                      const tagged_intersection<Index> &rec)
    -> bool {
  return extents_make_a_product(
      intersection_feature_extent<Index>(faces0, fe0, mel0, rec.object,
                                         rec.target),
      intersection_feature_extent<Index>(faces1, fe1, mel1, rec.object_other,
                                         rec.target_other));
}

/// Both sides of a contact expanded, and whether its fan is a product.
template <typename Index, typename Faces0, typename FE0, typename MEL0,
          typename Faces1, typename FE1, typename MEL1, typename Neighbors,
          typename Own, typename Other>
auto expand_intersection_sides(const Faces0 &faces0, const FE0 &fe0,
                               const MEL0 &mel0, const Faces1 &faces1,
                               const FE1 &fe1, const MEL1 &mel1,
                               const tagged_intersection<Index> &rec,
                               bool is_self, Neighbors &neighbors, Own &own,
                               Other &other) -> bool {
  own.clear();
  other.clear();
  expand_intersection_feature<Index>(faces0, fe0, mel0, rec.object, rec.target,
                                     is_self ? rec.object_other : Index(-1),
                                     neighbors, own);
  expand_intersection_feature<Index>(
      faces1, fe1, mel1, rec.object_other, rec.target_other,
      is_self ? rec.object : Index(-1), neighbors, other);
  return extents_make_a_product(own.size(), other.size());
}

} // namespace tf::intersect
