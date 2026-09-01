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
#include "../../topology/topo_type.hpp"
#include "./edge.hpp"
#include "./edges.hpp"
#include "./vertex.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// Both record endpoints present in the loop as created vertices.
template <typename Index, typename Loop>
auto loop_holds_both(const Loop &loop, Index a, Index b) -> bool {
  bool ha = false, hb = false;
  for (const auto &v : loop) {
    if (v.source != vertex_source::created)
      continue;
    ha = ha || v.id == a;
    hb = hb || v.id == b;
    if (ha && hb)
      return true;
  }
  return false;
}

/// Import the OTHER face's subdivision of a shared contact line.
///
/// Its loop is walked between the contact endpoints, emitting only the
/// pieces that do NOT lie on our own base loop — those are the loop's
/// own pairs. A failed walk leaves no trace.
template <typename Index, typename OurLoop, typename OtherLoop>
auto emit_other_subdivision(const OurLoop &our_loop,
                            const OtherLoop &other_loop, Index start_id,
                            Index end_id, short tag, short tag_other,
                            Index object, Index object_other,
                            tf::buffer<edge<Index>> &buf,
                            tf::buffer<std::int16_t> &covered) -> bool {
  using vsource = vertex_source;
  auto n = other_loop.size();
  std::size_t pos = n;
  bool end_present = false;
  for (std::size_t i = 0; i < n; ++i) {
    if (other_loop[i].source != vsource::created)
      continue;
    if (pos == n && other_loop[i].id == start_id)
      pos = i;
    if (other_loop[i].id == end_id)
      end_present = true;
    if (pos != n && end_present)
      break;
  }
  if (pos == n || !end_present)
    return false;

  auto rollback = buf.size();
  auto covered_rollback = covered.size();
  Index prev = start_id;
  pos = (pos + 1) % n;
  for (std::size_t step = 0; step < n - 1; ++step) {
    if (other_loop[pos].source != vsource::created) {
      buf.erase_till_end(buf.begin() + rollback);
      covered.erase_till_end(covered.begin() +
                             std::ptrdiff_t(covered_rollback));
      return false;
    }
    auto cur = other_loop[pos].id;
    auto ord = base_loop_edge_ordinal<Index>(our_loop, prev, cur);
    if (ord[0] == std::int16_t(-1))
      buf.push_back({tag, tag_other, object, object_other, prev, cur,
                     static_cast<Index>(buf.size()), std::int16_t(-1),
                     std::int16_t(-1)});
    else
      covered.push_back(ord[0]);
    if (cur == end_id)
      return true;
    prev = cur;
    pos = (pos + 1) % n;
  }
  buf.erase_till_end(buf.begin() + rollback);
  covered.erase_till_end(covered.begin() + std::ptrdiff_t(covered_rollback));
  return false;
}

/// Emit what only the records know; reject what the base loop states.
///
/// The face's whole base loop is already in the plane graph's edge set,
/// so a contact lying on it (A) is the loop's own and emits nothing. A
/// contact on the OTHER face's boundary (B) imports that face's
/// subdivision of the shared line: third-party cuts put created points
/// there that split our contact segment — points our own records cannot
/// know — and only the pieces off our loop emit. What remains is the
/// interior transversal segment.
///
/// An own-boundary contact whose endpoint was snipped from the loop
/// (antenna cleanup) is not covered by the loop pairs, so it degrades to
/// a direct interior chord.
template <typename Index, typename Record, typename AllLoops,
          typename Descriptors, typename ApplyToFace>
auto emit_plane_edge(const Record &r0, const Record &r1, Index face_size,
                     std::size_t this_loop_idx, const AllLoops &all_loops,
                     const Descriptors &descriptors,
                     const ApplyToFace &apply_to_face,
                     tf::buffer<edge<Index>> &buf,
                     tf::buffer<std::int16_t> &covered) -> void {
  auto tag = short(r0.tag);
  auto tag_other = short(r0.tag_other);

  if (r0.id == r1.id)
    return;

  auto &&our_loop = all_loops[this_loop_idx];

  if (on_same_boundary_edge(r0.target, r1.target, face_size) &&
      loop_holds_both<Index>(our_loop, r0.id, r1.id)) {
    // the original boundary walk owns the direction: it resolves the
    // forward order from the targets and the face winding, and its
    // stamp hands back every loop position the span covers. The
    // sub-edges it emits are transient here - the base loop states
    // this side, marked.
    const auto covered_rollback = covered.size();
    const auto transient = buf.size();
    const bool walked = try_expand_boundary<Index>(
        our_loop, r0.target, r1.target, face_size, r0.id, r1.id, tag,
        tag_other, r0.object, r0.object_other, buf,
        [&](Index, Index, std::size_t prev_pos) -> std::array<std::int16_t, 2> {
          covered.push_back(std::int16_t(prev_pos));
          return {std::int16_t(prev_pos), std::int16_t(0)};
        });
    buf.erase_till_end(buf.begin() + std::ptrdiff_t(transient));
    if (walked)
      return;
    covered.erase_till_end(covered.begin() +
                           std::ptrdiff_t(covered_rollback));
  }

  if (r0.target_other.label != tf::topo_type::face &&
      r1.target_other.label != tf::topo_type::face) {
    bool expanded = false;
    apply_to_face(r0.tag_other, r0.object_other, [&](const auto &other_face) {
      Index other_size = other_face.size();
      if (on_same_boundary_edge(r0.target_other, r1.target_other, other_size)) {
        auto idx = find_loop_index<Index>(descriptors, r0.tag_other,
                                          r0.object_other);
        if (idx != std::size_t(-1)) {
          Index start_id, end_id;
          if (resolve_boundary_walk(all_loops[idx], r0.target_other,
                                    r1.target_other, other_size, r0.id, r1.id,
                                    start_id, end_id))
            expanded = emit_other_subdivision<Index>(
                our_loop, all_loops[idx], start_id, end_id, tag, tag_other,
                r0.object, r0.object_other, buf, covered);
        }
      }
    });
    if (expanded)
      return;
  }

  auto ord = base_loop_edge_ordinal<Index>(our_loop, r0.id, r1.id);
  if (ord[0] != std::int16_t(-1)) {
    covered.push_back(ord[0]);
    return;
  }
  buf.push_back({tag, tag_other, r0.object, r0.object_other, r0.id, r1.id,
                 static_cast<Index>(buf.size()), std::int16_t(-1),
                 std::int16_t(-1)});
}

} // namespace tf::intersect::graph
