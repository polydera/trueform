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

#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/small_vector.hpp"
#include "../../topology/edge_id_in_face.hpp"
#include "../../topology/face_edge_neighbors.hpp"
#include "../../topology/vertex_id_in_face.hpp"
#include "./tagged_intersection.hpp"

namespace tf::exact {

namespace detail {

/// Expand the target_other side of an intersection record to neighbor
/// faces sharing the hit edge or vertex. For each expansion, calls
/// `emit(record)`.
template <typename Index, typename Faces, typename FE, typename MEL,
          typename Emit>
auto expand_other(const Faces &faces, const FE &fe, const MEL &mel,
                  tagged_intersection<Index> rec, Emit &&emit) {
  if (rec.target_other.label == tf::topo_type::edge) {
    emit(rec);
    auto N = faces[rec.object_other].size();
    Index e0 = faces[rec.object_other][rec.target_other.id];
    Index e1 =
        faces[rec.object_other]
             [tf::circular_increment<Index>(rec.target_other.id, Index(N))];

    auto &&link = mel[rec.object_other][rec.target_other.id];
    if (link.is_simple()) {
      Index n_face = link.face_peer;
      Index n_edge = tf::edge_id_in_face(e1, e0, faces[n_face]);
      auto n = rec;
      n.object_other = n_face;
      n.target_other.id = n_edge;
      emit(n);
    } else if (!link.is_manifold()) {
      tf::small_vector<Index, 5> neighbors;
      tf::face_edge_neighbors(fe, faces, rec.object_other, e0, e1,
                              std::back_inserter(neighbors));
      for (auto n_face : neighbors) {
        Index n_edge = tf::edge_id_in_face(e1, e0, faces[n_face]);
        auto n = rec;
        n.object_other = n_face;
        n.target_other.id = n_edge;
        emit(n);
      }
    }
  } else if (rec.target_other.label == tf::topo_type::vertex) {
    Index vid = faces[rec.object_other][rec.target_other.id];
    for (auto face_id : fe[vid]) {
      auto n = rec;
      n.object_other = face_id;
      n.target_other.id = tf::vertex_id_in_face<Index>(vid, faces[face_id]);
      emit(n);
    }
  } else {
    // face — no expansion needed
    emit(rec);
  }
}

} // namespace detail

/// Duplicate a single intersection record to all affected faces on both
/// meshes, and mirror each to the other mesh's perspective.
///
/// Expands the target side (mesh[tag]) and target_other side
/// (mesh[tag_other]) to neighbor faces sharing hit edges. Each expanded
/// record is pushed twice: original + mirrored.
template <typename Index, typename Faces0, typename FE0, typename MEL0,
          typename Faces1, typename FE1, typename MEL1>
auto duplicate_intersection(const Faces0 &faces0, const FE0 &fe0,
                            const MEL0 &mel0, const Faces1 &faces1,
                            const FE1 &fe1, const MEL1 &mel1,
                            tagged_intersection<Index> rec,
                            tf::buffer<tagged_intersection<Index>> &out) {

  auto push_both = [&](tagged_intersection<Index> r) {
    out.push_back(r);
    std::swap(r.tag, r.tag_other);
    std::swap(r.object, r.object_other);
    std::swap(r.target, r.target_other);
    out.push_back(r);
  };

  // Expand target side (mesh[tag]), then for each, expand target_other side
  // (mesh[tag_other]).
  auto expand_and_push = [&](tagged_intersection<Index> r) {
    detail::expand_other(faces1, fe1, mel1, r, push_both);
  };

  if (rec.target.label == tf::topo_type::edge) {
    expand_and_push(rec);
    auto N = faces0[rec.object].size();
    Index e0 = faces0[rec.object][rec.target.id];
    Index e1 = faces0[rec.object]
                     [tf::circular_increment<Index>(rec.target.id, Index(N))];

    auto &&link = mel0[rec.object][rec.target.id];
    if (link.is_simple()) {
      Index n_face = link.face_peer;
      Index n_edge = tf::edge_id_in_face(e1, e0, faces0[n_face]);
      auto n = rec;
      n.object = n_face;
      n.target.id = n_edge;
      expand_and_push(n);
    } else if (!link.is_manifold()) {
      tf::small_vector<Index, 5> neighbors;
      tf::face_edge_neighbors(fe0, faces0, rec.object, e0, e1,
                              std::back_inserter(neighbors));
      for (auto n_face : neighbors) {
        Index n_edge = tf::edge_id_in_face(e1, e0, faces0[n_face]);
        auto n = rec;
        n.object = n_face;
        n.target.id = n_edge;
        expand_and_push(n);
      }
    }
  } else if (rec.target.label == tf::topo_type::vertex) {
    Index vid = faces0[rec.object][rec.target.id];
    for (auto face_id : fe0[vid]) {
      auto n = rec;
      n.object = face_id;
      n.target.id = tf::vertex_id_in_face<Index>(vid, faces0[face_id]);
      expand_and_push(n);
    }
  } else {
    // face — no expansion on target side, just expand other side
    expand_and_push(rec);
  }
}

/// Returns a duplicator callable for use with `tf::generic_generate`.
/// Two-mesh overload: records with tag=0 use (form0, form1), tag=1 use
/// (form1, form0).
template <typename Policy0, typename Policy1>
auto make_duplicator(const tf::polygons<Policy0> &form0,
                     const tf::polygons<Policy1> &form1) {
  return [&](auto rec, auto &buffer) {
    if (rec.tag == 0)
      duplicate_intersection(form0.faces(), form0.face_membership(),
                             form0.manifold_edge_link(), form1.faces(),
                             form1.face_membership(),
                             form1.manifold_edge_link(), rec, buffer);
    else
      duplicate_intersection(form1.faces(), form1.face_membership(),
                             form1.manifold_edge_link(), form0.faces(),
                             form0.face_membership(),
                             form0.manifold_edge_link(), rec, buffer);
  };
}

/// Returns a duplicator callable for use with `tf::generic_generate`.
/// N-mesh overload: `forms[i]` must provide `.faces()`,
/// `.face_membership()`, and `.manifold_edge_link()`.
template <typename FormsRange> auto make_duplicator(const FormsRange &forms) {
  return [forms = tf::make_range(forms)](auto rec, auto &buffer) {
    auto &&f0 = forms[rec.tag];
    auto &&f1 = forms[rec.tag_other];
    duplicate_intersection(f0.faces(), f0.face_membership(),
                           f0.manifold_edge_link(), f1.faces(),
                           f1.face_membership(), f1.manifold_edge_link(), rec,
                           buffer);
  };
}

} // namespace tf::exact
