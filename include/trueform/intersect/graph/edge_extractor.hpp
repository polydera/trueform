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

#include "../../core/buffer.hpp"
#include "../../core/small_vector.hpp"
#include "../../topology/topo_type.hpp"
#include "../exact/tagged_intersection.hpp"
#include "./edge.hpp"
#include "./edges.hpp"

#include <algorithm>

namespace tf::intersect::graph {

/// Extracts intersection edges from a face's intersection records.
///
/// Routing is decided from call-time exact facts carried topologically —
/// never from computed coordinates. A group with any coplanar-flagged
/// record is a coplanar contact region; a group holding shared-vertex
/// records (both targets resolve to the same original vertex — delivered
/// by the duplicator to every self pair with contact) is a transversal
/// chord through that vertex; the rest routes by intersection count.
template <typename Index> class edge_extractor {
public:
  /// Extract edges for one face's intersection subrange.
  template <typename Subrange, typename AllLoops, typename AllSubranges,
            typename ApplyToFace>
  auto extract(const Subrange &subrange, Index face_size,
               std::size_t this_loop_idx, const AllLoops &all_loops,
               const AllSubranges &all_subranges,
               const ApplyToFace &apply_to_face,
               tf::buffer<edge<Index>> &buf) -> void {
    auto it = subrange.begin();
    auto end = subrange.end();
    while (it != end) {
      auto group_begin = it;
      it = std::find_if_not(it + 1, end, [&](const auto &r) {
        return r.tag_other == group_begin->tag_other &&
               r.object_other == group_begin->object_other;
      });
      auto n = it - group_begin;
      bool coplanar_pair = false;
      for (auto q = group_begin; q != it; ++q)
        if (q->flags & tf::intersect::coplanar_pair_flag) {
          coplanar_pair = true;
          break;
        }
      if (coplanar_pair) {
        apply_to_face(
            group_begin->tag_other, group_begin->object_other,
            [&](const auto &other_face) {
              Index other_size = other_face.size();
              extract_coplanar(group_begin, it, other_size, face_size,
                               this_loop_idx, all_loops, all_subranges,
                               apply_to_face, buf);
            });
        continue;
      }
      bool routed = false;
      if (group_begin->tag == group_begin->tag_other)
        routed = extract_shared(group_begin, it, face_size, this_loop_idx,
                                all_loops, all_subranges, apply_to_face, buf);
      if (routed)
        continue;
      if (n == 2) {
        emit_edge<Index>(group_begin[0], group_begin[1], face_size,
                         this_loop_idx, all_loops, all_subranges,
                         apply_to_face, buf);
      } else if (n > 2) {
        apply_to_face(
            group_begin->tag_other, group_begin->object_other,
            [&](const auto &other_face) {
              Index other_size = other_face.size();
              extract_coplanar(group_begin, it, other_size, face_size,
                               this_loop_idx, all_loops, all_subranges,
                               apply_to_face, buf);
            });
      }
    }
  }

  /// Route a transversal self-pair group holding shared-vertex records:
  /// the shared vertex pairs with each ordinary record (fold chord,
  /// corner-crossing chain); vertex-only groups are tangential contacts
  /// and stay silent. Returns false when the group has no shared-vertex
  /// records (caller falls back to count-based routing). Assumes
  /// convex faces: only then is the chord through the shared vertex the
  /// whole transversal intersection.
  template <typename Iterator, typename AllLoops, typename AllSubranges,
            typename ApplyToFace>
  auto extract_shared(Iterator group_begin, Iterator group_end,
                      Index face_size, std::size_t this_loop_idx,
                      const AllLoops &all_loops,
                      const AllSubranges &all_subranges,
                      const ApplyToFace &apply_to_face,
                      tf::buffer<edge<Index>> &buf) -> bool {
    bool vv_possible = false;
    for (auto q = group_begin; q != group_end; ++q)
      if (q->target.label == tf::topo_type::vertex &&
          q->target_other.label == tf::topo_type::vertex) {
        vv_possible = true;
        break;
      }
    if (!vv_possible)
      return false;
    bool handled = false;
    apply_to_face(group_begin->tag, group_begin->object, [&](const auto &face) {
      apply_to_face(
          group_begin->tag_other, group_begin->object_other,
          [&](const auto &other_face) {
            using rec_t = std::decay_t<decltype(*group_begin)>;
            tf::small_vector<const rec_t *, 8> shared_recs, ordinary_recs;
            auto seen = [](auto &v, Index id) {
              for (auto *r : v)
                if (Index(r->id) == id)
                  return true;
              return false;
            };
            for (auto q = group_begin; q != group_end; ++q) {
              bool shared_vv =
                  q->target.label == tf::topo_type::vertex &&
                  q->target_other.label == tf::topo_type::vertex &&
                  Index(face[q->target.id]) ==
                      Index(other_face[q->target_other.id]);
              auto &bucket = shared_vv ? shared_recs : ordinary_recs;
              if (!seen(bucket, Index(q->id)))
                bucket.push_back(&*q);
            }
            if (shared_recs.size() == 0)
              return;
            handled = true;
            for (auto *ord : ordinary_recs)
              // convex faces only: at most one shared vertex carries a
              // transversal chord, so the first shared record is the pair
              emit_edge<Index>(*ord, *shared_recs[0], face_size, this_loop_idx,
                               all_loops, all_subranges, apply_to_face, buf);
          });
    });
    return handled;
  }

private:
  struct coplanar_entry {
    Index edge_id;
    Index rec_idx; // offset from group_begin
  };

  /// Extract coplanar edges for an n>2 group.
  ///
  /// Groups records by target_other edge of the other face. Vertex contacts
  /// are duplicated to both adjacent edges. For each edge group with 2
  /// records where at least one is edge-type, emits via emit_edge.
  template <typename Iterator, typename AllLoops, typename AllSubranges,
            typename ApplyToFace>
  auto extract_coplanar(Iterator begin, Iterator end, Index other_poly_size,
                        Index face_size, std::size_t this_loop_idx,
                        const AllLoops &all_loops,
                        const AllSubranges &all_subranges,
                        const ApplyToFace &apply_to_face,
                        tf::buffer<edge<Index>> &buf) -> void {
    _work.clear();
    auto cp_start = buf.size();

    Index idx = 0;
    for (auto it = begin; it != end; ++it, ++idx) {
      if (it->target_other.label == tf::topo_type::face)
        continue;
      if (it->target_other.label == tf::topo_type::edge) {
        _work.push_back({Index(it->target_other.id), idx});
      } else if (it->target_other.label == tf::topo_type::vertex) {
        auto v = Index(it->target_other.id);
        auto prev_edge = (v - 1 + other_poly_size) % other_poly_size;
        _work.push_back({prev_edge, idx});
        _work.push_back({v, idx});
      }
    }

    if (_work.size() < 2)
      return;

    // rec_idx tie-break: group membership (exactly-2 emits) must not
    // depend on the parallel gather's record order.
    std::sort(_work.begin(), _work.end(),
              [](const auto &a, const auto &b) {
                return std::make_pair(a.edge_id, a.rec_idx) <
                       std::make_pair(b.edge_id, b.rec_idx);
              });
    // Multiple deliveries of the same shared vertex reach a pair once
    // per source record; identical (edge, point) entries are one
    // contact, not two.
    _work.erase_till_end(std::unique(
        _work.begin(), _work.end(), [&](const auto &a, const auto &b) {
          return a.edge_id == b.edge_id &&
                 Index((begin + a.rec_idx)->id) ==
                     Index((begin + b.rec_idx)->id);
        }));

    auto wit = _work.begin();
    auto wend = _work.end();
    while (wit != wend) {
      auto group_begin_w = wit;
      wit = std::find_if_not(wit + 1, wend, [&](const auto &e) {
        return e.edge_id == group_begin_w->edge_id;
      });
      if (wit - group_begin_w != 2)
        continue;
      auto &r0 = *(begin + group_begin_w->rec_idx);
      auto &r1 = *(begin + (group_begin_w + 1)->rec_idx);
      emit_edge<Index>(r0, r1, face_size, this_loop_idx, all_loops,
                       all_subranges, apply_to_face, buf);
    }

    // Coplanarity is known here, at emission: every edge this n>2 group
    // produced is a polygon-of-contact edge. The EE identity uses the stamp
    // to spot triples spanned by multiple contact edges.
    for (std::size_t i = cp_start; i < buf.size(); ++i)
      buf[i].from_coplanar = true;
  }

  tf::buffer<coplanar_entry> _work;
};

} // namespace tf::intersect::graph
