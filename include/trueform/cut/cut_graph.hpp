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
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/array_hash.hpp"
#include "../core/buffer.hpp"
#include "../core/hash_set.hpp"
#include "../core/memory.hpp"
#include "../core/views/mapped_range.hpp"
#include "../intersect/graph/intersection_graph.hpp"
#include "../intersect/graph/vertex.hpp"
#include "../topology/compare_faces.hpp"
#include "../topology/face_edge_neighbors.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/is_on_same_edge.hpp"
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

  template <typename Int>
  auto connectivity_per_face_edge(const tf::face_cuts<Index, Int> &fc) const {
    return tf::make_offset_block_range(fc.loops_buffer().offsets_buffer(),
                                       _connectivity);
  }
  auto coplanar_pairs() const { return tf::make_range(_coplanar_pairs); }
  auto coplanar_mask() const { return tf::make_range(_coplanar_mask); }

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
    _coplanar_mask.clear();
    _intersection_edges.clear();
    _ie_set.clear();
  }

  template<typename Int>
  auto build(const tf::face_cuts<Index, Int> &fc, Index n_ipts) -> void {
    clear();
    auto &src = fc.loops_buffer();
    if (!src.size())
      return;

    build_flat_ids(fc, n_ipts);
    build_connectivity(fc);
    build_coplanar_pairs(fc);
    build_intersection_edges(fc);
  }

  template <typename Int, typename Policy>
  auto build(const tf::face_cuts<Index, Int> &fc,
             const tf::intersection_graph<Index, Int> &ig,
             const tf::polygons<Policy> &polygons) -> void {
    clear();
    auto &src = fc.loops_buffer();
    if (!src.size())
      return;

    build_flat_ids(fc, static_cast<Index>(ig.points().size()));
    build_connectivity(fc);
    build_coplanar_pairs_self(fc);
    build_intersection_edges_self(polygons, ig, fc);
  }

private:
  template <typename Int>
  auto build_flat_ids(const tf::face_cuts<Index, Int> &fc, Index n_ipts) -> void {
    auto &src = fc.loops_buffer();
    auto tag_offs = fc.tag_offsets();
    auto n_tags = tag_offs.size() - 1;

    tf::core::std_vector<tf::hash_map<Index, Index>> maps(n_tags);
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
              d[i] = (s[i].source == source::created) ? s[i].id
                                                      : base + m[s[i].id];
        });

    _n_flat = tag_base[n_tags];
  }

  template <typename Int>
  auto build_connectivity(const tf::face_cuts<Index, Int> &fc) -> void {
    auto &src = fc.loops_buffer();
    auto flat_loops = tf::make_offset_block_range(src.offsets_buffer(),
                                                  tf::make_range(_flat_data));
    _fm.build(tf::make_faces(flat_loops), _n_flat, _flat_data.size());
    tf::topology::compute_face_link_per_edge(flat_loops, _fm, _connectivity);
  }

  template <bool WithSelf, typename Int>
  auto build_coplanar_pairs_impl(const tf::face_cuts<Index, Int> &fc) -> void {
    auto descs = fc.descriptors();
    auto loops = fc.loops();

    // Vertex equality is (source, id) — IDs are per-form for originals.
    // Wrap each vertex with an effective tag (created → -1, original →
    // form tag) so cross-tag comparison distinguishes same-id originals
    // from different forms.
    using vt = tf::intersect::graph::vertex<Index>;
    using key_t = std::pair<Index, vt>;
    auto with_tag = [](const auto &loop, Index tag) {
      return tf::make_mapped_range(loop, [tag](const vt &v) -> key_t {
        const Index eff =
            (v.source == tf::intersect::graph::vertex_source::created)
                ? Index(-1) : tag;
        return key_t{eff, v};
      });
    };

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
            if constexpr (WithSelf) {
              if (descs[nj].object == descs[li].object)
                continue;
            } else {
              if (descs[nj].tag == descs[li].tag)
                continue;
            }
            int cmp = tf::compare_faces(
                with_tag(loop, descs[Index(li)].tag),
                with_tag(loops[nj], descs[nj].tag));
            if (cmp != 0)
              out.push_back({Index(li), nj, cmp < 0});
          }
        });
  }

  template <typename Int>
  auto build_coplanar_pairs(const tf::face_cuts<Index, Int> &fc) -> void {
    return build_coplanar_pairs_impl<false>(fc);
  }

  template <typename Int>
  auto build_coplanar_pairs_self(const tf::face_cuts<Index, Int> &fc) -> void {
    return build_coplanar_pairs_impl<true>(fc);
  }

  template <typename Int>
  auto build_intersection_edges(const tf::face_cuts<Index, Int> &fc) -> void {
    auto descs = fc.descriptors();
    auto conn = connectivity_per_face_edge(fc);
    auto flat_loops = tf::make_offset_block_range(
        fc.loops_buffer().offsets_buffer(), tf::make_range(_flat_data));

    _coplanar_mask.allocate(flat_loops.size());
    tf::parallel_fill(_coplanar_mask, false);
    tf::parallel_for_each(_coplanar_pairs, [&](const auto &p) {
      _coplanar_mask[p.loop_a] = true;
      _coplanar_mask[p.loop_b] = true;
    });

    tf::generic_generate(tf::enumerate(tf::zip(flat_loops, conn)),
                         _intersection_edges, [&](const auto &item, auto &out) {
                           auto &&[li, tup] = item;
                           auto &&[loop, edge_neighbors] = tup;
                           if (_coplanar_mask[li])
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

  template <typename Int, typename Policy>
  auto build_intersection_edges_self(
      const tf::polygons<Policy> &polygons,
      const tf::intersection_graph<Index, Int> &ig,
      const tf::face_cuts<Index, Int> &fc) -> void {
    auto &&mel = polygons.manifold_edge_link();
    auto descs = fc.descriptors();
    auto loops = fc.loops();
    auto conn = connectivity_per_face_edge(fc);
    auto flat_loops = tf::make_offset_block_range(
        fc.loops_buffer().offsets_buffer(), tf::make_range(_flat_data));

    _coplanar_mask.allocate(flat_loops.size());
    tf::parallel_fill(_coplanar_mask, false);
    tf::parallel_for_each(_coplanar_pairs, [&](const auto &p) {
      _coplanar_mask[p.loop_a] = true;
      _coplanar_mask[p.loop_b] = true;
    });

    // Phase 1: copy all edges from intersection graph
    _intersection_edges.allocate(ig.edge_groups().size());
    tf::parallel_transform(ig.edge_groups(), _intersection_edges,
                           [](const auto &group) -> std::array<Index, 2> {
                             auto a = group[0].point_0, b = group[0].point_1;
                             return {std::min(a, b), std::max(a, b)};
                           });

    // Phase 2: boundary edges with new neighbors
    tf::generic_generate(
        tf::enumerate(tf::zip(flat_loops, conn, loops)), _intersection_edges,
        tf::small_vector<Index, 5>{},
        [&](const auto &item, auto &out, auto &neighbor_buf) {
          auto &&[li, tup] = item;
          auto &&[flat_loop, edge_neighbors, orig_loop] = tup;
          if (_coplanar_mask[li])
            return;
          auto my_object = descs[li].object;
          auto face = polygons.faces()[my_object];
          auto face_size = face.size();
          auto n = flat_loop.size();

          for (std::size_t ei = 0; ei < n; ++ei) {
            auto next = tf::circular_increment(ei, n);
            auto &v0 = orig_loop[ei];
            auto &v1 = orig_loop[next];

            if (!tf::is_on_same_edge(v0.sub_id, v1.sub_id, face_size))
              continue;

            // Determine which original edge this sits on
            auto edge_idx =
                (v0.sub_id.label == tf::topo_type::edge)   ? v0.sub_id.id
                : (v1.sub_id.label == tf::topo_type::edge) ? v1.sub_id.id
                                                           : v0.sub_id.id;

            auto &&link = mel[my_object][edge_idx];

            // Collect expected original neighbors for this edge
            neighbor_buf.clear();
            if (link.is_simple()) {
              neighbor_buf.push_back(link.face_peer);
            } else if (!link.is_manifold()) {
              Index e0 = face[edge_idx];
              Index e1 = face[tf::circular_increment(
                  static_cast<Index>(edge_idx),
                  static_cast<Index>(face_size))];
              tf::face_edge_neighbors(polygons.face_membership(),
                                      polygons.faces(), my_object, e0, e1,
                                      std::back_inserter(neighbor_buf));
            }
            // else: boundary edge — no expected neighbors

            // Check if any actual neighbor is NOT in expected set
            bool has_new_neighbor = false;
            for (auto nj : edge_neighbors[ei]) {
              auto nj_object = descs[nj].object;
              if (nj_object == my_object)
                continue;
              bool expected = false;
              for (auto exp : neighbor_buf)
                if (nj_object == exp) {
                  expected = true;
                  break;
                }
              if (!expected) {
                has_new_neighbor = true;
                break;
              }
            }

            if (has_new_neighbor) {
              auto a = flat_loop[ei], b = flat_loop[next];
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
  tf::buffer<bool> _coplanar_mask;
  tf::buffer<std::array<Index, 2>> _intersection_edges;
  tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> _ie_set;
};

} // namespace tf
