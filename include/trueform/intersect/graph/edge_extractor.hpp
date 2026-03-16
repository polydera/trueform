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
#include "../../topology/topo_type.hpp"
#include "./edge.hpp"
#include "./edges.hpp"

#include <algorithm>

namespace tf::intersect::graph {

/// Extracts intersection edges from a face's intersection records.
///
/// Handles n==2 (standard pair), n>2 coplanar (group by target_other edge),
/// and all-VV skip (shared vertices). Holds reusable scratch buffers to
/// avoid allocations in the hot path.
template <typename Index> class edge_extractor {
public:
  /// Extract edges for one face's intersection subrange.
  template <typename Subrange, typename AllLoops, typename AllSubranges,
            typename GetFace>
  auto extract(const Subrange &subrange, Index face_size,
               const AllLoops &all_loops, const AllSubranges &all_subranges,
               const GetFace &get_face,
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
      if (n == 2) {
        emit_edge<Index>(group_begin[0], group_begin[1], face_size,
                         all_loops, all_subranges, get_face, buf);
      } else if (n > 2) {
        Index other_size =
            get_face(group_begin->tag_other, group_begin->object_other)
                .size();
        extract_coplanar(group_begin, it, other_size, face_size,
                         all_loops, all_subranges, get_face, buf);
      }
    }
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
            typename GetFace>
  auto extract_coplanar(Iterator begin, Iterator end, Index other_poly_size,
                        Index face_size, const AllLoops &all_loops,
                        const AllSubranges &all_subranges,
                        const GetFace &get_face,
                        tf::buffer<edge<Index>> &buf) -> void {
    _work.clear();

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

    std::sort(_work.begin(), _work.end(),
              [](const auto &a, const auto &b) {
                return a.edge_id < b.edge_id;
              });

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
      emit_edge<Index>(r0, r1, face_size, all_loops, all_subranges, get_face,
                       buf);
    }
  }

  tf::buffer<coplanar_entry> _work;
};

} // namespace tf::intersect::graph
