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

#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../intersect/graph/vertex.hpp"
#include "../topology/compare_faces.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/structures/compute_face_link_per_edge.hpp"
#include "./face_cuts.hpp"

#include <vector>

namespace tf {

/// @ingroup cut
/// @brief A pair of coplanar face cuts from different meshes.
template <typename Index> struct coplanar_pair {
  Index loop_a, loop_b;
  bool opposing; // true if reversed winding
};

/// @ingroup cut
/// @brief Per-edge connectivity graph built from face cuts.
///
/// Flattens vertex<Index> loops to compact integer IDs, builds
/// face_membership, then compute_face_link_per_edge for per-loop
/// per-edge neighbor lookup.
template <typename Index> class cut_graph {
  using vertex_t = intersect::graph::vertex<Index>;
  using source = intersect::graph::vertex_source;

public:
  auto connectivity() const { return tf::make_range(_connectivity); }

  auto clear() -> void {
    _flat_data.clear();
    _fm.clear();
    _connectivity.clear();
  }

  auto build(const tf::face_cuts<Index> &fc, Index n_ipts) -> void {
    clear();
    auto &src = fc.loops_buffer();
    if (!src.size())
      return;

    auto tag_offs = fc.tag_offsets();
    auto n_tags = tag_offs.size() - 1;

    // Phase 1: per-tag hash maps — collect unique original vertex IDs
    std::vector<tf::hash_map<Index, Index>> maps(n_tags);
    auto tag_loops =
        tf::make_offset_block_range(tag_offs, fc.loops());
    tf::parallel_for_each(tf::enumerate(tag_loops), [&](auto pair) {
      auto &&[tag, loops] = pair;
      auto &m = maps[tag];
      for (auto &&loop : loops)
        for (auto &&v : loop)
          if (v.source == source::original)
            if (m.find(v.id) == m.end())
              m[v.id] = Index(m.size());
    });

    // Phase 2: prefix-sum per-tag counts → global bases
    tf::buffer<Index> tag_base;
    tag_base.allocate(n_tags + 1);
    tag_base[0] = 0;
    for (std::size_t t = 0; t < n_tags; ++t)
      tag_base[t + 1] = tag_base[t] + Index(maps[t].size());
    Index N = tag_base[n_tags];

    // Phase 3: parallel per tag — write flat IDs
    _flat_data.allocate(src.data_buffer().size());
    auto tag_src = tf::make_offset_block_range(
        tag_offs, tf::make_offset_block_range(src.offsets_buffer(),
                                              src.data_buffer()));
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
              d[i] = (s[i].source == source::created) ? N + s[i].id
                                                      : base + m[s[i].id];
        });

    // Phase 4: face_membership + compute_face_link_per_edge
    auto flat_loops = tf::make_offset_block_range(src.offsets_buffer(),
                                                  tf::make_range(_flat_data));
    _fm.build(tf::make_faces(flat_loops), N + n_ipts, _flat_data.size());
    tf::topology::compute_face_link_per_edge(flat_loops, _fm, _connectivity);
  }

  auto find_coplanar_pairs(const tf::face_cuts<Index> &fc)
      -> tf::buffer<coplanar_pair<Index>> {
    auto descs = fc.descriptors();
    auto loops = fc.loops();
    tf::buffer<coplanar_pair<Index>> pairs;

    tf::generic_generate(
        tf::enumerate(tf::zip(loops, connectivity())), pairs,
        [&](const auto &item, auto &out) {
          auto &&[li, tup] = item;
          auto &&[loop, edge_neighbors] = tup;
          // check first edge only
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

    return pairs;
  }

private:
  tf::buffer<Index> _flat_data;
  tf::face_membership<Index> _fm;
  tf::offset_block_buffer<Index, Index> _connectivity;
};

} // namespace tf
