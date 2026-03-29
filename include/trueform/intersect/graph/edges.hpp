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
#include "../../topology/is_on_same_edge.hpp"
#include "../../topology/topo_type.hpp"
#include "./edge.hpp"
#include "./vertex.hpp"

#include <algorithm>
#include <utility>

namespace tf::intersect::graph {

/// Check if two targets lie on the same boundary segment of a face.
///
/// vertex-vertex: consecutive vertices (share a face edge)
/// edge-edge: same edge index
/// vertex-edge: vertex is an endpoint of the edge
template <typename Target, typename Index>
auto on_same_boundary_edge(const Target &t0, const Target &t1, Index face_size)
    -> bool {
  return tf::is_on_same_edge(t0, t1, face_size);
}

/// Walk a loop forward from start_id to end_id, emitting sub-edges.
/// Returns true if intermediates were found and sub-edges emitted.
template <typename Index, typename Loop>
auto emit_boundary_sub_edges(const Loop &loop, Index start_id, Index end_id,
                             short tag, short tag_other, Index object,
                             Index object_other, tf::buffer<edge<Index>> &buf)
    -> bool {
  auto n = loop.size();
  std::size_t pos = n;
  for (std::size_t i = 0; i < n; ++i) {
    if (loop[i].source == vertex_source::created && loop[i].id == start_id) {
      pos = i;
      break;
    }
  }
  if (pos == n)
    return false;

  Index prev = start_id;
  pos = (pos + 1) % n;
  for (std::size_t step = 0; step < n - 1; ++step) {
    auto cur = loop[pos].id;
    if (loop[pos].source == vertex_source::created && cur == end_id) {
      if (prev == start_id)
        return false;
      buf.push_back({tag, tag_other, object, object_other, prev, cur,
                     static_cast<Index>(buf.size())});
      return true;
    }
    buf.push_back({tag, tag_other, object, object_other, prev, cur,
                   static_cast<Index>(buf.size())});
    prev = cur;
    pos = (pos + 1) % n;
  }
  return false;
}

/// Determine walk direction and expand boundary if intermediates exist.
///
/// For two targets on the same boundary edge, determines the forward
/// direction (following the face winding) and walks the loop to find
/// intermediate intersection points.
template <typename Index, typename Target, typename Loop>
auto try_expand_boundary(const Loop &loop, const Target &t0, const Target &t1,
                         Index face_size, Index id0, Index id1, short tag,
                         short tag_other, Index object, Index object_other,
                         tf::buffer<edge<Index>> &buf) -> bool {
  if (!on_same_boundary_edge(t0, t1, face_size))
    return false;

  auto start_id = id0, end_id = id1;

  auto v = tf::topo_type::vertex;
  auto e = tf::topo_type::edge;

  if (t0.label == v && t1.label == v) {
    // vertex-vertex: forward is v0 → v1 where v1 = circular_increment(v0)
    if (tf::circular_increment(Index(t1.id), face_size) == Index(t0.id))
      std::swap(start_id, end_id);
  } else if (t0.label == e && t1.label == e) {
    // edge-edge: both on same edge, loop has them in parametric order.
    // Scan for whichever appears first — that's the start.
    auto n = loop.size();
    for (std::size_t i = 0; i < n; ++i) {
      if (loop[i].source != vertex_source::created)
        continue;
      if (loop[i].id == id0)
        break; // id0 first → default order is correct
      if (loop[i].id == id1) {
        std::swap(start_id, end_id);
        break;
      }
    }
  } else {
    // vertex-edge: the vertex at the START of the edge comes first
    auto edge_id = (t0.label == e) ? Index(t0.id) : Index(t1.id);
    auto vert_id = (t0.label == v) ? Index(t0.id) : Index(t1.id);
    auto vert_rec_id = (t0.label == v) ? id0 : id1;
    auto edge_rec_id = (t0.label == e) ? id0 : id1;
    if (vert_id == edge_id) {
      start_id = vert_rec_id;
      end_id = edge_rec_id;
    } else {
      start_id = edge_rec_id;
      end_id = vert_rec_id;
    }
  }

  return emit_boundary_sub_edges<Index>(loop, start_id, end_id, tag, tag_other,
                                        object, object_other, buf);
}

/// Find loop index for a (tag, object) pair via binary search on subranges.
template <typename Index, typename AllSubranges>
auto find_loop_index(const AllSubranges &subranges, Index tag, Index object)
    -> std::size_t {
  auto key = std::make_pair(tag, object);
  auto it =
      std::lower_bound(subranges.begin(), subranges.end(), key,
                       [](const auto &sr, const auto &k) {
                         return std::make_pair(sr[0].tag, sr[0].object) < k;
                       });
  if (it == subranges.end())
    return std::size_t(-1);
  auto &&sr = *it;
  if (std::make_pair(sr[0].tag, sr[0].object) != key)
    return std::size_t(-1);
  return static_cast<std::size_t>(it - subranges.begin());
}

/// Emit an edge unless it lies on our own boundary.
/// If it lies on the other face's boundary, expand to catch intermediates.
template <typename Index, typename Record, typename AllLoops,
          typename AllSubranges, typename ApplyToFace>
auto emit_edge(const Record &r0, const Record &r1, Index face_size,
               const AllLoops &all_loops, const AllSubranges &all_subranges,
               const ApplyToFace &apply_to_face, tf::buffer<edge<Index>> &buf)
    -> void {
  auto tag = short(r0.tag);
  auto tag_other = short(r0.tag_other);

  // Edge on our own boundary → skip. The base loop already has these points.
  if (on_same_boundary_edge(r0.target, r1.target, face_size))
    return;

  if (r0.target_other.label != tf::topo_type::face &&
      r1.target_other.label != tf::topo_type::face) {
    bool expanded = false;
    apply_to_face(r0.tag_other, r0.object_other, [&](const auto &other_face) {
      Index other_size = other_face.size();
      if (on_same_boundary_edge(r0.target_other, r1.target_other, other_size)) {
        auto idx = find_loop_index<Index>(all_subranges, r0.tag_other,
                                          r0.object_other);
        if (idx != std::size_t(-1))
          expanded = try_expand_boundary<Index>(
              all_loops[idx], r0.target_other, r1.target_other, other_size,
              r0.id, r1.id, tag, tag_other, r0.object, r0.object_other, buf);
      }
    });
    if (expanded)
      return;
  }

  buf.push_back({tag, tag_other, r0.object, r0.object_other, r0.id, r1.id,
                 static_cast<Index>(buf.size())});
}

/// Extract edges for a subrange, expanding boundary edges using base loops.
template <typename Index, typename Subrange, typename AllLoops,
          typename AllSubranges, typename ApplyToFace>
auto extract_edges_with_expansion(const Subrange &subrange, Index face_size,
                                  std::size_t this_loop_idx,
                                  const AllLoops &all_loops,
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
    if (it - group_begin == 2)
      emit_edge<Index>(group_begin[0], group_begin[1], face_size, this_loop_idx,
                       all_loops, all_subranges, apply_to_face, buf);
  }
}

} // namespace tf::intersect::graph
