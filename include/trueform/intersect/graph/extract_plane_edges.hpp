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

#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../../exact/meta.hpp"
#include "../../topology/topo_type.hpp"
#include "../records/tagged_intersection.hpp"
#include "./edge.hpp"
#include "./plane_graph_edges.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::intersect::graph {

/// One distinct contact point of a group, with the record that states
/// its topology.
template <typename Index, typename Int, typename Record> struct chain_node {
  tf::point<Int, 3> pt;
  Index id;
  const Record *rec;
};

/// Chain a group's contacts along the pair's intersection line.
///
/// The records of a non-coplanar pair are contacts on the ONE line the
/// two planes meet in, so their points are collinear by construction
/// and a single coordinate orders them: the axis of largest extent,
/// which no two distinct points on the line share. Identity is the
/// point id — a record's id is a canonical point name, one name per
/// position, so distinct ids are distinct points, and the first record
/// of an id states its topology (a point reached through several
/// contacts of the same pair repeats the id, never the position).
/// Consecutive points bound the pieces of the contact; each goes
/// through @ref tf::intersect::graph::emit_plane_edge, which rejects
/// what the base loops already say.
template <typename Index, typename Int, typename Record, typename Iterator,
          typename AllLoops, typename Descriptors, typename ApplyToFace,
          typename GetPoint>
auto emit_collinear_chain(
    Iterator group_begin, Iterator group_end, Index face_size,
    std::size_t this_loop_idx, const AllLoops &all_loops,
    const Descriptors &descriptors, const ApplyToFace &apply_to_face,
    const GetPoint &get_point, tf::buffer<chain_node<Index, Int, Record>> &work,
    tf::buffer<edge<Index>> &buf, tf::buffer<std::int16_t> &covered) -> void {
  work.clear();
  for (auto q = group_begin; q != group_end; ++q) {
    bool seen = false;
    for (const auto &node : work)
      seen = seen || node.id == Index(q->id);
    if (!seen)
      work.push_back(
          {get_point(std::int16_t(-1), Index(q->id)), Index(q->id), &*q});
  }
  if (work.size() < 2)
    return;

  using T1 = typename tf::exact::meta<Int>::T1;
  int axis = 0;
  T1 widest(0);
  for (int a = 0; a < 3; ++a) {
    Int lo = work[0].pt[a], hi = lo;
    for (const auto &node : work) {
      lo = node.pt[a] < lo ? node.pt[a] : lo;
      hi = hi < node.pt[a] ? node.pt[a] : hi;
    }
    const T1 extent = T1(hi) - lo;
    if (widest < extent) {
      widest = extent;
      axis = a;
    }
  }
  const int next_axis = (axis + 1) % 3;
  const int last_axis = (axis + 2) % 3;
  std::sort(work.begin(), work.end(),
            [axis, next_axis, last_axis](const auto &a, const auto &b) {
              if (a.pt[axis] != b.pt[axis])
                return a.pt[axis] < b.pt[axis];
              if (a.pt[next_axis] != b.pt[next_axis])
                return a.pt[next_axis] < b.pt[next_axis];
              if (a.pt[last_axis] != b.pt[last_axis])
                return a.pt[last_axis] < b.pt[last_axis];
              return a.id < b.id;
            });
  for (std::size_t k = 1; k < work.size(); ++k)
    emit_plane_edge<Index>(*work[k - 1].rec, *work[k].rec, face_size,
                           this_loop_idx, all_loops, descriptors,
                           apply_to_face, buf, covered);
}

/// Transversal self-pair group holding shared-vertex records: the
/// shared vertex pairs with each ordinary record (fold chord, corner
/// crossing).
template <typename Index, typename Iterator, typename AllLoops,
          typename Descriptors, typename ApplyToFace>
auto extract_shared_plane_edges(Iterator group_begin, Iterator group_end,
                                Index face_size, std::size_t this_loop_idx,
                                const AllLoops &all_loops,
                                const Descriptors &descriptors,
                                const ApplyToFace &apply_to_face,
                                tf::buffer<edge<Index>> &buf,
                                tf::buffer<std::int16_t> &covered) -> bool {
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
            bool shared_vv = q->target.label == tf::topo_type::vertex &&
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
            emit_plane_edge<Index>(*ord, *shared_recs[0], face_size,
                                   this_loop_idx, all_loops, descriptors,
                                   apply_to_face, buf, covered);
        });
  });
  return handled;
}

/// Route one face's intersection records into plane-graph edges.
///
/// A coplanar pair pools both faces into ONE plane carrier and both
/// their base loops enter that carrier's edge set wholesale, so the
/// contact region is already stated and such a group emits nothing.
/// What remains is the transversal routing: shared-vertex chords,
/// 2-record contour segments, and the collinear chain of a group of
/// more than two — a non-coplanar pair meets in one line, so its
/// records are contacts along it. That case carries a face whose
/// lattice vertices are collinear: it has no plane of its own, so a
/// pair of it and the face it lies in is never classified coplanar and
/// all of its vertex/edge contacts arrive in one group.
///
/// @param chain the collinear chain's scratch, reused across the calls
///        of one parallel block.
template <typename Index, typename Int, typename Subrange, typename AllLoops,
          typename Descriptors, typename ApplyToFace, typename GetPoint>
auto extract_plane_edges(
    const Subrange &subrange, Index face_size, std::size_t this_loop_idx,
    const AllLoops &all_loops, const Descriptors &descriptors,
    const ApplyToFace &apply_to_face, const GetPoint &get_point,
    tf::buffer<
        chain_node<Index, Int, tf::intersect::tagged_intersection<Index>>>
        &chain,
    tf::buffer<edge<Index>> &buf, tf::buffer<std::int16_t> &covered) -> void {
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
    if (coplanar_pair)
      continue;
    bool routed = false;
    if (group_begin->tag == group_begin->tag_other)
      routed = extract_shared_plane_edges<Index>(
          group_begin, it, face_size, this_loop_idx, all_loops, descriptors,
          apply_to_face, buf, covered);
    if (routed)
      continue;
    if (n == 2)
      emit_plane_edge<Index>(group_begin[0], group_begin[1], face_size,
                             this_loop_idx, all_loops, descriptors,
                             apply_to_face, buf, covered);
    else if (n > 2)
      emit_collinear_chain<Index, Int>(group_begin, it, face_size,
                                       this_loop_idx, all_loops, descriptors,
                                       apply_to_face, get_point, chain, buf,
                                       covered);
  }
}

} // namespace tf::intersect::graph
