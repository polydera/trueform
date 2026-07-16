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

#include <array>

namespace tf::intersect {

namespace impl {

template <typename Index, typename Faces, typename FE, typename MEL,
          typename Emit>
auto expand_other(const Faces &faces, const FE &fe, const MEL &mel,
                  tagged_intersection<Index> rec, bool is_self,
                  Emit &&emit) {
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
      if (is_self && face_id == rec.object)
        continue;
      auto n = rec;
      n.object_other = face_id;
      n.target_other.id = tf::vertex_id_in_face<Index>(vid, faces[face_id]);
      emit(n);
    }
  } else {
    emit(rec);
  }
}

template <typename Index, typename Faces0, typename FE0, typename MEL0,
          typename Faces1, typename FE1, typename MEL1>
auto duplicate_intersection_impl(const Faces0 &faces0, const FE0 &fe0,
                                 const MEL0 &mel0, const Faces1 &faces1,
                                 const FE1 &fe1, const MEL1 &mel1,
                                 tagged_intersection<Index> rec,
                                 tf::buffer<tagged_intersection<Index>> &out,
                                 bool is_self, Index sentinel_base = 0) {
  const Index orig_object = rec.object;
  const Index orig_object_other = rec.object_other;
  tf::small_vector<std::array<Index, 2>, 16> pairs_done;

  // Any record delivered to a self pair whose faces share vertices
  // delivers those vertices too: the shared vertex lies on the pair's
  // intersection (it is in both faces), and it is the chord endpoint the
  // shared-mask suppression never records. Pure id scan — no geometry.
  // The id is a sentinel (sentinel_base + global vertex id) resolved to a
  // point after the generate pass, upstream of dedup.
  auto deliver_shared_vertices = [&](const tagged_intersection<Index> &r) {
    if (!is_self || sentinel_base == 0 || r.object == r.object_other)
      return;
    for (const auto &p : pairs_done)
      if (p[0] == r.object && p[1] == r.object_other)
        return;
    pairs_done.push_back({r.object, r.object_other});
    auto &&f0 = faces0[r.object];
    auto &&f1 = faces1[r.object_other];
    Index n0 = Index(f0.size());
    Index n1 = Index(f1.size());
    for (Index i = 0; i < n0; ++i)
      for (Index j = 0; j < n1; ++j) {
        if (Index(f0[i]) != Index(f1[j]))
          continue;
        // The record fans across the vertex's whole face fan (one copy
        // per incident face, anchored to the contact pair) — the same
        // identity synchronization ordinary vertex-target records get
        // from the fan expansion: every incident loop substitutes the
        // corner with the SAME created point, so cut and uncut faces
        // agree on the vertex's identity. Fan copies landing in
        // non-contact groups are vertex-only and stay silent.
        const Index vid = Index(f0[i]);
        for (auto g : fe0[vid]) {
          const Index partner =
              Index(g) == r.object ? r.object_other : r.object;
          if (Index(g) == partner)
            continue;
          tagged_intersection<Index> v;
          v.tag = r.tag;
          v.tag_other = r.tag_other;
          v.object = Index(g);
          v.object_other = partner;
          v.target = {tf::vertex_id_in_face<Index>(vid, faces0[g]),
                      tf::topo_type::vertex};
          v.target_other =
              {tf::vertex_id_in_face<Index>(vid, faces0[partner]),
               tf::topo_type::vertex};
          v.id = sentinel_base + vid;
          v.flags = 0;
          out.push_back(v);
          std::swap(v.tag, v.tag_other);
          std::swap(v.object, v.object_other);
          std::swap(v.target, v.target_other);
          out.push_back(v);
        }
        break;
      }
  };

  auto push_both = [&](tagged_intersection<Index> r) {
    // A set coplanar flag describes the emitting pair; clear it on copies
    // rewritten to another pair.
    if (r.object != orig_object || r.object_other != orig_object_other)
      r.flags = 0;
    deliver_shared_vertices(r);
    out.push_back(r);
    std::swap(r.tag, r.tag_other);
    std::swap(r.object, r.object_other);
    std::swap(r.target, r.target_other);
    out.push_back(r);
  };

  auto expand_and_push = [&](tagged_intersection<Index> r) {
    impl::expand_other(faces1, fe1, mel1, r, is_self, push_both);
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
      if (is_self && face_id == rec.object_other)
        continue;
      auto n = rec;
      n.object = face_id;
      n.target.id = tf::vertex_id_in_face<Index>(vid, faces0[face_id]);
      expand_and_push(n);
    }
  } else {
    expand_and_push(rec);
  }
}

} // namespace impl

/// Two-mesh duplicate_intersection (is_self = false).
template <typename Index, typename Faces0, typename FE0, typename MEL0,
          typename Faces1, typename FE1, typename MEL1>
auto duplicate_intersection(const Faces0 &faces0, const FE0 &fe0,
                            const MEL0 &mel0, const Faces1 &faces1,
                            const FE1 &fe1, const MEL1 &mel1,
                            tagged_intersection<Index> rec,
                            tf::buffer<tagged_intersection<Index>> &out) {
  impl::duplicate_intersection_impl(faces0, fe0, mel0, faces1, fe1, mel1, rec,
                                    out, false);
}

/// Self-intersection duplicate_intersection (is_self = true).
/// `sentinel_base` (= point count before duplication) keys the sentinel
/// ids of delivered shared-vertex records.
template <typename Index, typename Faces, typename FE, typename MEL>
auto duplicate_intersection_self(const Faces &faces, const FE &fe,
                                 const MEL &mel,
                                 tagged_intersection<Index> rec,
                                 tf::buffer<tagged_intersection<Index>> &out,
                                 Index sentinel_base = 0) {
  impl::duplicate_intersection_impl(faces, fe, mel, faces, fe, mel, rec, out,
                                    true, sentinel_base);
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
/// N-mesh overload.
template <typename FormsRange> auto make_duplicator(const FormsRange &forms) {
  return [forms = tf::make_range(forms)](auto rec, auto &buffer) {
    auto &&f0 = forms[rec.tag];
    auto &&f1 = forms[rec.tag_other];
    duplicate_intersection(
        f0.faces(), f0.face_membership(), f0.manifold_edge_link(), f1.faces(),
        f1.face_membership(), f1.manifold_edge_link(), rec, buffer);
  };
}

/// Returns a self-intersection duplicator callable.
template <typename Policy>
auto make_self_duplicator(const tf::polygons<Policy> &form) {
  return [&](auto rec, auto &buffer) {
    duplicate_intersection_self(form.faces(), form.face_membership(),
                                form.manifold_edge_link(), rec, buffer);
  };
}

} // namespace tf::intersect
