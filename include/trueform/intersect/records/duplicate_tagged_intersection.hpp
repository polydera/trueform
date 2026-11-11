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

#include "../../core/none.hpp"
#include "../../core/small_vector.hpp"
#include "../../topology/topo_type.hpp"
#include "../../topology/vertex_id_in_face.hpp"
#include "./expand_intersection_sides.hpp"
#include "./point_delivery.hpp"
#include "./shared_vertex_claim.hpp"
#include "./tagged_intersection.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::intersect {

/// THE COUNT AND THE WRITE ARE ONE PRODUCER. The duplication's shape is a
/// function of the record, the topology and the gate alone, so the same walk
/// answers how many copies a record makes and where they go: the counting
/// sinks take the first pass, the span sinks the second, and the exact sizes
/// the prefixes between them state are written once into disjoint slices.
template <typename Index> struct duplicate_counting_sink {
  std::size_t count = 0;
  auto push_back(const tagged_intersection<Index> &) -> void { ++count; }
};

template <typename Index> struct duplicate_span_sink {
  tagged_intersection<Index> *at;
  auto push_back(const tagged_intersection<Index> &rec) -> void {
    *at++ = rec;
  }
};

template <typename Index> struct delivery_counting_sink {
  std::size_t count = 0;
  auto push_back(const point_delivery<Index> &) -> void { ++count; }
};

template <typename Index> struct delivery_span_sink {
  point_delivery<Index> *at;
  auto push_back(const point_delivery<Index> &rec) -> void { *at++ = rec; }
};

template <typename Index> struct claim_counting_sink {
  std::size_t count = 0;
  auto push_back(const shared_vertex_claim<Index> &) -> void { ++count; }
};

template <typename Index> struct claim_span_sink {
  shared_vertex_claim<Index> *at;
  auto push_back(const shared_vertex_claim<Index> &claim) -> void {
    *at++ = claim;
  }
};

/// The fan's block-local scratch: the two feature expansions, a
/// non-manifold edge's faces, the gate's answer for one side of the
/// product, and which instances of each side a kept pair already names.
template <typename Index, typename Side> struct intersection_fan_scratch {
  tf::small_vector<feature_instance<Index>, 16> own;
  tf::small_vector<feature_instance<Index>, 16> other;
  tf::small_vector<Index, 5> neighbors;
  tf::small_vector<Side, 16> other_sides;
  tf::small_vector<char, 16> own_named;
  tf::small_vector<char, 16> other_named;
};

/// The fan's three currencies, stated in ONE walk.
///
/// The pair currency is one record per (face, face) the contact's two
/// features stand on, both orientations, for the pairs the gate admits —
/// and each of those records already NAMES the point at its own face, so
/// it carries that face's identity too. The identity currency is
/// therefore the DIFFERENCE: an expansion instance no admitted pair
/// names, and nothing else. A record the gate never asks about names
/// every instance, so its difference is empty.
///
/// The claim currency is what the self family leaves for
/// @ref state_shared_vertex_fans: one `{tag, face, vertex}` per shared
/// vertex of a pair it states, whose fan is not this walk's fact to
/// state — it is a function of the claim alone, and any number of pairs
/// make the same claim.
///
/// Passing @ref tf::none_t for the gate admits every pair — what the self
/// family needs, its emissions being the discovery site of its own
/// shared-vertex deliveries.
template <typename Index, typename Faces0, typename FE0, typename MEL0,
          typename Faces1, typename FE1, typename MEL1, typename Sink,
          typename Deliveries, typename Claims, typename Gate, typename Scratch>
auto duplicate_intersection_impl(const Faces0 &faces0, const FE0 &fe0,
                                 const MEL0 &mel0, const Faces1 &faces1,
                                 const FE1 &fe1, const MEL1 &mel1,
                                 const tagged_intersection<Index> &rec,
                                 Sink &out, Deliveries &deliveries,
                                 Claims &claims, bool is_self,
                                 Index sentinel_base, const Gate &gate,
                                 Scratch &scratch) {
  // Any record delivered to a self pair whose faces share vertices states
  // those vertices too: the shared vertex lies on the pair's intersection
  // (it is in both faces), and it is the chord endpoint the shared-mask
  // suppression never records. Pure id scan — no geometry. The id is a
  // sentinel (sentinel_base + global vertex id) resolved to a point after
  // the generate pass, upstream of dedup. The vertex's own fan is the
  // claim's fact, not this pair's.
  auto deliver_shared_vertices = [&](const tagged_intersection<Index> &r) {
    if (!is_self || sentinel_base == 0 || r.object == r.object_other)
      return;
    auto &&f0 = faces0[r.object];
    auto &&f1 = faces1[r.object_other];
    Index n0 = Index(f0.size());
    Index n1 = Index(f1.size());
    for (Index i = 0; i < n0; ++i)
      for (Index j = 0; j < n1; ++j) {
        if (Index(f0[i]) != Index(f1[j]))
          continue;
        const Index vid = Index(f0[i]);
        tagged_intersection<Index> v;
        v.tag = r.tag;
        v.tag_other = r.tag_other;
        v.object = r.object;
        v.object_other = r.object_other;
        v.target = {tf::vertex_id_in_face<Index>(vid, f0),
                    tf::topo_type::vertex};
        v.target_other = {tf::vertex_id_in_face<Index>(vid, f1),
                          tf::topo_type::vertex};
        v.id = sentinel_base + vid;
        out.push_back(v);
        std::swap(v.tag, v.tag_other);
        std::swap(v.object, v.object_other);
        std::swap(v.target, v.target_other);
        out.push_back(v);
        claims.push_back({r.tag, r.object, vid});
        break;
      }
  };

  auto push_both = [&](tagged_intersection<Index> r) {
    deliver_shared_vertices(r);
    out.push_back(r);
    std::swap(r.tag, r.tag_other);
    std::swap(r.object, r.object_other);
    std::swap(r.target, r.target_other);
    out.push_back(r);
  };

  auto &own = scratch.own;
  auto &other = scratch.other;
  const bool prunable = expand_intersection_sides<Index>(
      faces0, fe0, mel0, faces1, fe1, mel1, rec, is_self, scratch.neighbors,
      own, other);

  auto push_pair = [&](const feature_instance<Index> &a,
                       const feature_instance<Index> &b) {
    auto r = rec;
    r.object = a.object;
    r.target.id = a.target_id;
    r.object_other = b.object;
    r.target_other.id = b.target_id;
    push_both(r);
  };

  constexpr bool gated = !std::is_same_v<Gate, tf::none_t>;
  if (!gated || !prunable) {
    for (const auto &a : own)
      for (const auto &b : other)
        push_pair(a, b);
    return;
  }
  if constexpr (gated) {
    auto &other_sides = scratch.other_sides;
    auto &own_named = scratch.own_named;
    auto &other_named = scratch.other_named;
    other_sides.clear();
    own_named.assign(own.size(), char(0));
    other_named.assign(other.size(), char(0));
    for (const auto &b : other)
      other_sides.push_back(gate.side_of(rec.tag_other, b.object));
    for (std::size_t i = 0; i < own.size(); ++i) {
      const auto a_side = gate.side_of(rec.tag, own[i].object);
      for (std::size_t j = 0; j < other.size(); ++j) {
        if (!gate.keeps(a_side, other_sides[j]))
          continue;
        push_pair(own[i], other[j]);
        own_named[i] = char(1);
        other_named[j] = char(1);
      }
    }
    for (std::size_t i = 0; i < own.size(); ++i)
      if (!own_named[i])
        deliveries.push_back({rec.tag,
                              own[i].object,
                              {own[i].target_id, rec.target.label},
                              rec.id});
    for (std::size_t j = 0; j < other.size(); ++j)
      if (!other_named[j])
        deliveries.push_back({rec.tag_other,
                              other[j].object,
                              {other[j].target_id, rec.target_other.label},
                              rec.id});
  }
}

/// Two-mesh duplicate_intersection (is_self = false).
template <typename Index, typename Faces0, typename FE0, typename MEL0,
          typename Faces1, typename FE1, typename MEL1, typename Sink,
          typename Deliveries, typename Claims, typename Gate, typename Scratch>
auto duplicate_intersection(const Faces0 &faces0, const FE0 &fe0,
                            const MEL0 &mel0, const Faces1 &faces1,
                            const FE1 &fe1, const MEL1 &mel1,
                            const tagged_intersection<Index> &rec, Sink &out,
                            Deliveries &deliveries, Claims &claims,
                            const Gate &gate, Scratch &scratch) {
  duplicate_intersection_impl(faces0, fe0, mel0, faces1, fe1, mel1, rec, out,
                              deliveries, claims, false, Index(0), gate,
                              scratch);
}

/// Self-intersection duplicate_intersection (is_self = true).
/// `sentinel_base` (= point count before duplication) keys the sentinel
/// ids of delivered shared-vertex records.
template <typename Index, typename Faces, typename FE, typename MEL,
          typename Sink, typename Deliveries, typename Claims, typename Scratch>
auto duplicate_intersection_self(const Faces &faces, const FE &fe,
                                 const MEL &mel,
                                 const tagged_intersection<Index> &rec,
                                 Sink &out, Deliveries &deliveries,
                                 Claims &claims, Index sentinel_base,
                                 Scratch &scratch) {
  duplicate_intersection_impl(faces, fe, mel, faces, fe, mel, rec, out,
                              deliveries, claims, true, sentinel_base, tf::none,
                              scratch);
}

} // namespace tf::intersect
