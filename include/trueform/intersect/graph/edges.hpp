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
#include <array>
#include <cstddef>
#include <cstdint>
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

/// `{ordinal, sub_ordinal}` for the sub-edge `(a, b)` iff `a` and `b` are
/// consecutive created vertices of `loop` — i.e. the sub-edge is a
/// base-loop edge. The earlier of the two in loop order is the start, so
/// `ordinal` is its position (mirroring the edge-edge case of
/// @ref try_expand_boundary). Otherwise `{-1, -1}` (interior).
template <typename Index, typename Loop>
auto base_loop_edge_ordinal(const Loop &loop, Index a, Index b)
    -> std::array<std::int16_t, 2> {
  const std::size_t m = loop.size();
  for (std::size_t i = 0; i < m; ++i) {
    if (loop[i].source != vertex_source::created || loop[i].id != a)
      continue;
    const std::size_t nxt = (i + 1) % m;
    const std::size_t prv = (i + m - 1) % m;
    if (loop[nxt].source == vertex_source::created && loop[nxt].id == b)
      return {static_cast<std::int16_t>(i), std::int16_t(0)};
    if (loop[prv].source == vertex_source::created && loop[prv].id == b)
      return {static_cast<std::int16_t>(prv), std::int16_t(0)};
    break;
  }
  return {std::int16_t(-1), std::int16_t(-1)};
}

/// Walk `loop` forward from `start_id` to `end_id`, emitting one sub-edge
/// per step. `stamp(prev, cur, prev_pos)` returns the emitted edge's
/// `{ordinal, sub_ordinal}`. Returns true once `end_id` is reached.
///
/// Either endpoint may be missing from the loop: a record names a point
/// the loop does not have to carry. A failed walk must leave no trace —
/// the emitted sub-edges would carry ORIGINAL vertex ids, which
/// downstream stages dereference as intersection-point ids — so the
/// buffer rolls back and the caller degrades the contact to a direct
/// chord.
template <typename Index, typename Loop, typename Stamp>
auto emit_boundary_sub_edges(const Loop &loop, Index start_id, Index end_id,
                             short tag, short tag_other, Index object,
                             Index object_other, tf::buffer<edge<Index>> &buf,
                             const Stamp &stamp) -> bool {
  auto n = loop.size();
  std::size_t pos = n;
  bool end_present = false;
  for (std::size_t i = 0; i < n; ++i) {
    if (loop[i].source != vertex_source::created)
      continue;
    if (pos == n && loop[i].id == start_id)
      pos = i;
    if (loop[i].id == end_id)
      end_present = true;
    if (pos != n && end_present)
      break;
  }
  if (pos == n || !end_present)
    return false;

  auto rollback = buf.size();
  Index prev = start_id;
  std::size_t prev_pos = pos;
  pos = (pos + 1) % n;
  for (std::size_t step = 0; step < n - 1; ++step) {
    // The legitimate walk stays on one base edge, where only created
    // points lie between the two contacts. Hitting an original corner
    // means the on-edge occurrence of end_id was snipped and only a
    // far pinch occurrence remains — walking on would reach it around
    // the loop, emitting mesh-vertex ids as intersection points.
    if (loop[pos].source != vertex_source::created)
      break;
    auto cur = loop[pos].id;
    auto stamped = stamp(prev, cur, prev_pos);
    buf.push_back({tag, tag_other, object, object_other, prev, cur,
                   static_cast<Index>(buf.size()), stamped[0], stamped[1]});
    if (cur == end_id)
      return true;
    prev = cur;
    prev_pos = pos;
    pos = (pos + 1) % n;
  }
  buf.erase_till_end(buf.begin() + rollback);
  return false;
}

/// Resolve the forward walk direction for a contact on `loop`'s boundary.
///
/// For two targets on the same boundary edge, picks `start_id`/`end_id`
/// so that walking the loop forward follows the face winding. Returns
/// false (no walk) when the targets are not on the same boundary edge.
template <typename Index, typename Target, typename Loop>
auto resolve_boundary_walk(const Loop &loop, const Target &t0, const Target &t1,
                           Index face_size, Index id0, Index id1,
                           Index &start_id, Index &end_id) -> bool {
  if (!on_same_boundary_edge(t0, t1, face_size))
    return false;

  start_id = id0;
  end_id = id1;

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
  return true;
}

/// Expand a contact lying on `loop`'s boundary into per-edge sub-edges.
///
/// Resolves the forward direction, then walks `loop`, emitting each
/// sub-edge with `stamp(prev, cur, prev_pos) -> {ordinal, sub_ordinal}`.
template <typename Index, typename Target, typename Loop, typename Stamp>
auto try_expand_boundary(const Loop &loop, const Target &t0, const Target &t1,
                         Index face_size, Index id0, Index id1, short tag,
                         short tag_other, Index object, Index object_other,
                         tf::buffer<edge<Index>> &buf, const Stamp &stamp)
    -> bool {
  Index start_id, end_id;
  if (!resolve_boundary_walk(loop, t0, t1, face_size, id0, id1, start_id,
                             end_id))
    return false;
  return emit_boundary_sub_edges<Index>(loop, start_id, end_id, tag, tag_other,
                                        object, object_other, buf, stamp);
}

/// Find loop index for a (tag, object) pair via binary search on the face
/// carrier's descriptors, which the deliveries left ordered by that key.
template <typename Index, typename Descriptors>
auto find_loop_index(const Descriptors &descriptors, Index tag, Index object)
    -> std::size_t {
  auto key = std::make_pair(tag, object);
  auto it = std::lower_bound(
      descriptors.begin(), descriptors.end(), key,
      [](const auto &d, const auto &k) {
        return std::make_pair(Index(d.tag), Index(d.object)) < k;
      });
  if (it == descriptors.end())
    return std::size_t(-1);
  if (std::make_pair(Index(it->tag), Index(it->object)) != key)
    return std::size_t(-1);
  return static_cast<std::size_t>(it - descriptors.begin());
}
} // namespace tf::intersect::graph
