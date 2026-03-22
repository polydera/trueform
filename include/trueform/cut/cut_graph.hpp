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

#include "../core/algorithm/circular_increment.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/array_hash.hpp"
#include "../core/buffer.hpp"
#include "../core/hash_set.hpp"
#include "../intersect/graph/vertex.hpp"
#include "../topology/compare_faces.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/structures/compute_face_link_per_edge.hpp"
#include "./face_cuts.hpp"

#include <vector>

#include "tbb/parallel_sort.h"

namespace tf {

/// @ingroup cut
/// @brief Per-edge connectivity graph built from face cuts.
///
/// Flattens vertex<Index> loops to compact integer IDs, builds
/// face_membership, then compute_face_link_per_edge for per-loop
/// per-edge neighbor lookup. Also detects coplanar pairs and
/// intersection edges.
template <typename Index> class cut_graph {
  using vertex_t = intersect::graph::vertex<Index>;
  using source = intersect::graph::vertex_source;

public:
  struct coplanar_pair {
    Index loop_a, loop_b;
    bool opposing;
  };

  auto connectivity_per_face_edge(const tf::face_cuts<Index> &fc) const {
    return tf::make_offset_block_range(fc.loops_buffer().offsets_buffer(),
                                       _connectivity);
  }
  auto coplanar_pairs() const { return tf::make_range(_coplanar_pairs); }
  auto intersection_edges() const {
    return tf::make_range(_intersection_edges);
  }
  auto is_intersection_edge(Index flat_v0, Index flat_v1) const -> bool {
    return _ie_set.count(
               {std::min(flat_v0, flat_v1), std::max(flat_v0, flat_v1)}) > 0;
  }
  auto is_intersection_edge(const vertex_t &v0, const vertex_t &v1) const
      -> bool {
    if (v0.source != source::created || v1.source != source::created)
      return false;
    return is_intersection_edge(v0.id, v1.id);
  }

  auto clear() -> void {
    _flat_data.clear();
    _fm.clear();
    _connectivity.clear();
    _coplanar_pairs.clear();
    _intersection_edges.clear();
    _ie_set.clear();
  }

  auto build(const tf::face_cuts<Index> &fc, Index n_ipts) -> void {
    clear();
    auto &src = fc.loops_buffer();
    if (!src.size())
      return;

    build_flat_ids(fc, n_ipts);
    build_connectivity(fc);
    build_coplanar_pairs(fc);
    build_intersection_edges(fc);
  }

private:
  auto build_flat_ids(const tf::face_cuts<Index> &fc, Index n_ipts) -> void {
    auto &src = fc.loops_buffer();
    auto tag_offs = fc.tag_offsets();
    auto n_tags = tag_offs.size() - 1;

    std::vector<tf::hash_map<Index, Index>> maps(n_tags);
    auto tag_loops = tf::make_offset_block_range(tag_offs, fc.loops());
    tf::parallel_for_each(tf::enumerate(tag_loops), [&](auto pair) {
      auto &&[tag, loops] = pair;
      auto &m = maps[tag];
      for (auto &&loop : loops)
        for (auto &&v : loop)
          if (v.source == source::original)
            if (m.find(v.id) == m.end())
              m[v.id] = Index(m.size());
    });

    tf::buffer<Index> tag_base;
    tag_base.allocate(n_tags + 1);
    tag_base[0] = n_ipts;
    for (std::size_t t = 0; t < n_tags; ++t)
      tag_base[t + 1] = tag_base[t] + Index(maps[t].size());

    _flat_data.allocate(src.data_buffer().size());
    auto tag_src = tf::make_offset_block_range(
        tag_offs,
        tf::make_offset_block_range(src.offsets_buffer(), src.data_buffer()));
    auto tag_dst = tf::make_offset_block_range(
        tag_offs,
        tf::make_offset_block_range(src.offsets_buffer(), _flat_data));
    tf::parallel_for_each(
        tf::enumerate(tf::zip(tag_src, tag_dst)), [&](auto pair) {
          auto &&[tag, tup] = pair;
          auto &&[src_loops, dst_loops] = tup;
          auto base = tag_base[tag];
          auto &m = maps[tag];
          for (auto &&[s, d] : tf::zip(src_loops, dst_loops))
            for (std::size_t i = 0; i < s.size(); ++i)
              d[i] = (s[i].source == source::created)
                         ? s[i].id
                         : base + m[s[i].id];
        });

    _n_flat = tag_base[n_tags];
  }

  auto build_connectivity(const tf::face_cuts<Index> &fc) -> void {
    auto &src = fc.loops_buffer();
    auto flat_loops = tf::make_offset_block_range(src.offsets_buffer(),
                                                  tf::make_range(_flat_data));
    _fm.build(tf::make_faces(flat_loops), _n_flat, _flat_data.size());
    tf::topology::compute_face_link_per_edge(flat_loops, _fm, _connectivity);
  }

  auto build_coplanar_pairs(const tf::face_cuts<Index> &fc) -> void {
    auto descs = fc.descriptors();
    auto loops = fc.loops();

    tf::generic_generate(
        tf::enumerate(tf::zip(loops, connectivity_per_face_edge(fc))),
        _coplanar_pairs, [&](const auto &item, auto &out) {
          auto &&[li, tup] = item;
          auto &&[loop, edge_neighbors] = tup;
          if (edge_neighbors.size() == 0 || edge_neighbors[0].size() == 0)
            return;
          for (auto nj : edge_neighbors[0]) {
            if (nj <= Index(li))
              continue;
            if (descs[nj].tag == descs[li].tag)
              continue;
            int cmp = tf::compare_faces(loop, loops[nj]);
            if (cmp != 0)
              out.push_back({Index(li), nj, cmp < 0});
          }
        });
  }

  auto build_intersection_edges(const tf::face_cuts<Index> &fc) -> void {
    auto descs = fc.descriptors();
    auto conn = connectivity_per_face_edge(fc);
    auto flat_loops = tf::make_offset_block_range(
        fc.loops_buffer().offsets_buffer(), tf::make_range(_flat_data));

    tf::buffer<bool> is_coplanar;
    is_coplanar.allocate(flat_loops.size());
    tf::parallel_fill(is_coplanar, false);
    tf::parallel_for_each(_coplanar_pairs, [&](const auto &p) {
      is_coplanar[p.loop_a] = true;
      is_coplanar[p.loop_b] = true;
    });

    tf::generic_generate(tf::enumerate(tf::zip(flat_loops, conn)),
                         _intersection_edges, [&](const auto &item, auto &out) {
                           auto &&[li, tup] = item;
                           auto &&[loop, edge_neighbors] = tup;
                           if (is_coplanar[li])
                             return;
                           auto my_tag = descs[li].tag;
                           auto n = loop.size();
                           for (std::size_t ei = 0; ei < n; ++ei) {
                             bool has_cross_tag = false;
                             for (auto nj : edge_neighbors[ei]) {
                               if (descs[nj].tag != my_tag) {
                                 has_cross_tag = true;
                                 break;
                               }
                             }
                             if (has_cross_tag) {
                               auto next = tf::circular_increment(ei, n);
                               auto a = loop[ei], b = loop[next];
                               out.push_back({std::min(a, b), std::max(a, b)});
                             }
                           }
                         });

    tbb::parallel_sort(_intersection_edges.begin(), _intersection_edges.end());
    _intersection_edges.erase_till_end(
        std::unique(_intersection_edges.begin(), _intersection_edges.end()));

    for (auto &&e : _intersection_edges)
      _ie_set.insert(e);
  }

  Index _n_flat = 0;
  tf::buffer<Index> _flat_data;
  tf::face_membership<Index> _fm;
  tf::offset_block_buffer<Index, Index> _connectivity;
  tf::buffer<coplanar_pair> _coplanar_pairs;
  tf::buffer<std::array<Index, 2>> _intersection_edges;
  tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> _ie_set;
};

} // namespace tf
