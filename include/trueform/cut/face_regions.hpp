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

#include "../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/none.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/array_hash.hpp"
#include "../core/contiguous_index_hash_map.hpp"
#include "../core/index_hash_map.hpp"
#include "../core/points_buffer.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/slice.hpp"
#include "../exact/projection_axes.hpp"
#include "../intersect/graph/face_descriptor.hpp"
#include "../intersect/graph/intersection_graph.hpp"
#include "../intersect/graph/vertex.hpp"
#include "../topology/face_split_by_edges.hpp"
#include <algorithm>

namespace tf {

/// Splits all intersected faces using intersection_graph output,
/// emitting the RAW region structure: boundary walks plus hole walks,
/// no hole patching. The structure-native sibling of @ref tf::face_cuts
/// — a region with holes stays (boundary, holes) so a constrained
/// triangulation can consume the walks as constraints directly, and a
/// hole's interior piece is itself a region (its own loop).
///
/// Layout: `loops()` holds every region boundary walk grouped per face
/// and `holes()` the hole-walk pool, both flat CSRs. `loop_holes()`
/// maps each loop to the ids of its bounding holes — sparse blocks for
/// the rare hole-bearing loops behind a dense O(1) per-loop index.
template <typename Index, typename Int = tf::exact::int32> class face_regions {
  using vertex_t = intersect::graph::vertex<Index>;
  using desc_t = intersect::graph::face_descriptor<Index>;
  using source = intersect::graph::vertex_source;

public:
  auto loops() const { return tf::make_range(_loops); }
  auto loops_buffer() const
      -> const tf::offset_block_buffer<Index, vertex_t> & {
    return _loops;
  }
  auto holes() const { return tf::make_range(_holes); }
  auto holes_buffer() const
      -> const tf::offset_block_buffer<Index, vertex_t> & {
    return _holes;
  }
  /// Per loop, the ids of its bounding holes (into `holes()`) — a
  /// random-access range sized like `loops()`. Sparse underneath:
  /// hole-bearing loops are rare, so only they own a block of hole ids
  /// and a dense per-loop index (-1 = no holes) resolves in O(1).
  auto loop_holes() const {
    const Index *data = _loop_hole_blocks.data_buffer().begin();
    const Index *offs = _loop_hole_blocks.offsets_buffer().begin();
    const Index *index = _loop_hole_index.begin();
    return tf::make_mapped_range(
        tf::make_sequence_range(static_cast<Index>(loops().size())),
        [data, offs, index](Index li) {
          const Index b = index[std::size_t(li)];
          if (b == Index(-1))
            return tf::make_range(data, data);
          return tf::make_range(data + offs[std::size_t(b)],
                                data + offs[std::size_t(b) + 1]);
        });
  }
  auto descriptors() const { return tf::make_range(_descriptors); }
  auto tag_offsets() const { return tf::make_range(_tag_offsets); }

  /// Cut-and-consumed faces of form `tag`: faces whose records produced
  /// no loops. They are cut faces with no surviving geometry —
  /// consumers deciding cut/uncut must treat them as cut, never as
  /// untouched originals.
  auto deleted(Index tag) const {
    return tf::slice(tf::make_range(_deleted),
                     std::size_t(_deleted_offsets[std::size_t(tag)]),
                     std::size_t(_deleted_offsets[std::size_t(tag) + 1]));
  }
  auto deleted_offsets() const { return tf::make_range(_deleted_offsets); }

  auto clear() {
    _loops.clear();
    _holes.clear();
    _loop_hole_blocks.clear();
    _block_loops.clear();
    _loop_hole_index.clear();
    _descriptors.clear();
    _tag_offsets.clear();
    _deleted.clear();
    _deleted_offsets.clear();
  }

  template <typename ApplyToFace, typename GetMeshPoint>
  auto build(const tf::intersection_graph<Index, Int> &ig,
             const ApplyToFace &apply_to_face,
             const GetMeshPoint &get_mesh_point) -> void {
    clear();
    auto descs = ig.descriptors();
    auto all_loops = ig.loops();
    auto all_edges = ig.edges();
    auto pts = ig.points();
    if (descs.size() == 0) {
      tf::buffer<desc_t> no_deleted;
      build_deleted(no_deleted, ig.tag_offsets().size() - 1);
      return build_tag_offsets(ig.tag_offsets().size() - 1);
    }

    tf::buffer<desc_t> deleted_descs;
    tf::buffer<std::array<Index, 2>> rebase; // {offsets start, data delta}
    tf::buffer<std::array<Index, 2>> lh_entries; // {loop, hole}
    auto task = [&](auto &&range, local_t &local) {
      auto apply_to_face_f = apply_to_face;
      auto get_mesh_point_f = get_mesh_point;
      for (const auto &[desc, loop, edges] : range) {
        if (desc.tag == Index(-1))
          continue;
        if (loop.size() < 3) {
          local.deleted.push_back(desc);
          continue;
        }
        auto get_point = [&, &desc =
                                 desc](const vertex_t &v) -> tf::point<Int, 3> {
          if (v.source == source::created)
            return pts[v.id];
          return get_mesh_point_f(desc.tag, v.id);
        };
        dedup_edges(local, edges);
        apply_to_face_f(desc.tag, desc.object,
                        [&, &desc = desc, &loop = loop](const auto &face) {
                          auto n_loops = cut_face(local, loop, face, get_point);
                          if (n_loops == 0)
                            local.deleted.push_back(desc);
                          for (Index i = 0; i < n_loops; ++i)
                            local.descs.push_back(desc);
                        });
      }
    };
    auto agg = [&](const local_t &local, const tf::none_t &) {
      aggregate(local, deleted_descs, rebase, lh_entries);
    };
    tf::blocked_reduce_sequenced_aggregate(tf::zip(descs, all_loops, all_edges),
                                           tf::none, local_t{}, task, agg);
    if (_loops.offsets_buffer().size()) {
      rebase.push_back(
          {static_cast<Index>(_loops.offsets_buffer().size()), Index(0)});
      tf::parallel_for_each(
          tf::make_sequence_range(rebase.size() - 1), [&](std::size_t b) {
            const Index delta = rebase[b][1];
            if (delta == Index(0))
              return;
            for (Index i = rebase[b][0]; i < rebase[b + 1][0]; ++i)
              _loops.offsets_buffer()[std::size_t(i)] += delta;
          });
      _loops.offsets_buffer().push_back(
          static_cast<Index>(_loops.data_buffer().size()));
    }
    if (_holes.offsets_buffer().size())
      _holes.offsets_buffer().push_back(
          static_cast<Index>(_holes.data_buffer().size()));
    build_loop_holes(lh_entries);

    build_tag_offsets(ig.tag_offsets().size() - 1);
    build_deleted(deleted_descs, ig.tag_offsets().size() - 1);
  }

private:
  struct hash_t {
    auto operator()(const vertex_t &v) const {
      return hash(std::array<Index, 2>{Index(v.source), v.id});
    }
    tf::array_hash<Index, 2> hash;
  };

  struct local_t {
    tf::buffer<Index> offsets;
    tf::buffer<vertex_t> vertices;
    tf::buffer<Index> hole_offsets;
    tf::buffer<vertex_t> hole_vertices;
    tf::buffer<std::array<Index, 2>> lh_entries; // {local loop, local hole}
    tf::buffer<desc_t> descs;
    tf::buffer<desc_t> deleted;
    tf::buffer<std::array<Index, 2>> edge_buf;
    // identity -> contiguous local id for face_split_by_edges — the
    // face_cutter mechanism, verbatim
    tf::index_hash_map<vertex_t, Index, hash_t> ihm;
    tf::buffer<Index> base_loop;
    tf::buffer<Index> edges;
    tf::points_buffer<Int, 2> points;
    tf::face_split_by_edges<Index, Int> fs;
  };

  template <typename Edges>
  static auto dedup_edges(local_t &local, const Edges &edges) -> void {
    local.edge_buf.clear();
    for (auto &&e : edges) {
      local.edge_buf.push_back(
          {std::min(e.point_0, e.point_1), std::max(e.point_0, e.point_1)});
    }
    std::sort(local.edge_buf.begin(), local.edge_buf.end());
    local.edge_buf.erase_till_end(
        std::unique(local.edge_buf.begin(), local.edge_buf.end()));
    local.edge_buf.erase_till_end(
        std::remove_if(local.edge_buf.begin(), local.edge_buf.end(),
                       [](auto e) { return e[0] == e[1]; }));
  }

  /// Fast paths first: an uncut face emits its walk, a single edge with
  /// both endpoints on the walk splits it in two. Everything else runs
  /// the planar split in this face's exact projection.
  template <typename Loop, typename Face, typename GetPoint>
  static auto cut_face(local_t &local, const Loop &loop, const Face &face,
                       const GetPoint &get_point) -> Index {
    if (local.edge_buf.size() == 0) {
      emit_loop(loop, local.offsets, local.vertices);
      return 1;
    }
    if (local.edge_buf.size() == 1) {
      auto [p0, p1] = find_pair_in_loop(loop, local.edge_buf[0][0],
                                        local.edge_buf[0][1]);
      if (p0 >= 0 && p1 >= 0) {
        split_at(loop, p0, p1, local.offsets, local.vertices);
        return 2;
      }
    }
    auto axes = tf::exact::projection_axes(
        get_point(vertex_t{source::original,
                           Index(face[0]),
                           {0, tf::topo_type::vertex}}),
        get_point(vertex_t{source::original,
                           Index(face[1]),
                           {1, tf::topo_type::vertex}}),
        get_point(vertex_t{source::original,
                           Index(face[2]),
                           {2, tf::topo_type::vertex}}));
    return split_generic(local, loop, axes, get_point);
  }

  /// Sequenced aggregation stays O(appended bytes): the loop offsets go
  /// in raw (one memcpy) with a {start, delta} record per block, and a
  /// parallel post-pass rebases them after the reduce — the per-element
  /// serial loop was the scaling bottleneck at 52k loops. Hole offsets
  /// stay per-element: hole-carrying blocks are rare and short.
  auto aggregate(const local_t &local, tf::buffer<desc_t> &deleted_descs,
                 tf::buffer<std::array<Index, 2>> &rebase,
                 tf::buffer<std::array<Index, 2>> &lh_entries) -> void {
    tf::core::append(local.deleted, deleted_descs);
    if (local.offsets.size() == 0)
      return;
    auto offset = static_cast<Index>(_loops.data_buffer().size());
    auto old_off = _loops.offsets_buffer().size();
    rebase.push_back({static_cast<Index>(old_off), offset});
    tf::core::append(local.offsets, _loops.offsets_buffer());
    tf::core::append(local.vertices, _loops.data_buffer());
    tf::core::append(local.descs, _descriptors);

    auto hole_base = static_cast<Index>(_holes.offsets_buffer().size());
    auto hole_off = static_cast<Index>(_holes.data_buffer().size());
    auto old_h = _holes.offsets_buffer().size();
    _holes.offsets_buffer().reallocate(old_h + local.hole_offsets.size());
    for (std::size_t i = 0; i < local.hole_offsets.size(); ++i)
      _holes.offsets_buffer()[old_h + i] = local.hole_offsets[i] + hole_off;
    tf::core::append(local.hole_vertices, _holes.data_buffer());

    const auto loop_base = static_cast<Index>(old_off);
    for (const auto &e : local.lh_entries)
      lh_entries.push_back({e[0] + loop_base, e[1] + hole_base});
  }

  /// Group the sparse {loop, hole} entries into per-loop blocks — they
  /// arrive grouped and ordered by loop because aggregation is
  /// sequenced — then build the dense per-loop index in parallel.
  auto build_loop_holes(const tf::buffer<std::array<Index, 2>> &lh_entries)
      -> void {
    for (std::size_t i = 0; i < lh_entries.size();) {
      const Index li = lh_entries[i][0];
      _loop_hole_blocks.offsets_buffer().push_back(
          static_cast<Index>(_loop_hole_blocks.data_buffer().size()));
      _block_loops.push_back(li);
      while (i < lh_entries.size() && lh_entries[i][0] == li)
        _loop_hole_blocks.data_buffer().push_back(lh_entries[i++][1]);
    }
    if (_loop_hole_blocks.offsets_buffer().size())
      _loop_hole_blocks.offsets_buffer().push_back(
          static_cast<Index>(_loop_hole_blocks.data_buffer().size()));
    _loop_hole_index.allocate(loops().size());
    tf::parallel_fill(_loop_hole_index, Index(-1));
    tf::parallel_for_each(
        tf::make_sequence_range(_block_loops.size()), [&](std::size_t b) {
          _loop_hole_index[std::size_t(_block_loops[b])] =
              static_cast<Index>(b);
        });
  }

  template <typename Loop>
  static auto find_pair_in_loop(const Loop &loop, Index id0, Index id1)
      -> std::pair<Index, Index> {
    Index p0 = -1, p1 = -1;
    for (Index i = 0; i < static_cast<Index>(loop.size()); ++i) {
      if (loop[i].source != source::created)
        continue;
      if (loop[i].id == id0)
        p0 = i;
      else if (loop[i].id == id1)
        p1 = i;
      if (p0 >= 0 && p1 >= 0)
        return {p0, p1};
    }
    return {p0, p1};
  }

  template <typename Loop>
  static auto emit_loop(const Loop &loop, tf::buffer<Index> &offsets,
                        tf::buffer<vertex_t> &vertices) -> void {
    offsets.push_back(static_cast<Index>(vertices.size()));
    for (const auto &v : loop)
      vertices.push_back(v);
  }

  template <typename Loop>
  static auto split_at(const Loop &loop, Index pos0, Index pos1,
                       tf::buffer<Index> &offsets,
                       tf::buffer<vertex_t> &vertices) -> void {
    if (pos0 > pos1)
      std::swap(pos0, pos1);
    auto n = static_cast<Index>(loop.size());

    offsets.push_back(static_cast<Index>(vertices.size()));
    for (Index i = pos0; i <= pos1; ++i)
      vertices.push_back(loop[i]);

    offsets.push_back(static_cast<Index>(vertices.size()));
    for (Index i = pos1; i < n; ++i)
      vertices.push_back(loop[i]);
    for (Index i = 0; i <= pos0; ++i)
      vertices.push_back(loop[i]);
  }

  template <typename Loop, typename Axes, typename GetPoint>
  static auto split_generic(local_t &local, const Loop &loop, const Axes &axes,
                            const GetPoint &get_point) -> Index {
    local.ihm.clear();
    tf::make_contiguous_index_hash_map(loop, local.ihm, Index(0));
    for (const auto &e : local.edge_buf) {
      auto v0 = vertex_t{source::created, e[0], {0, tf::topo_type::none}};
      auto v1 = vertex_t{source::created, e[1], {0, tf::topo_type::none}};
      if (local.ihm.f().find(v0) == local.ihm.f().end()) {
        local.ihm.kept_ids().push_back(v0);
        local.ihm.f()[v0] = static_cast<Index>(local.ihm.kept_ids().size() - 1);
      }
      if (local.ihm.f().find(v1) == local.ihm.f().end()) {
        local.ihm.kept_ids().push_back(v1);
        local.ihm.f()[v1] = static_cast<Index>(local.ihm.kept_ids().size() - 1);
      }
    }

    const auto &kept = local.ihm.kept_ids();
    local.base_loop.clear();
    for (const auto &v : loop)
      local.base_loop.push_back(local.ihm.f()[v]);
    local.edges.clear();
    for (const auto &e : local.edge_buf) {
      local.edges.push_back(
          local.ihm.f()[vertex_t{source::created, e[0], {0, tf::topo_type::none}}]);
      local.edges.push_back(
          local.ihm.f()[vertex_t{source::created, e[1], {0, tf::topo_type::none}}]);
    }

    local.points.clear();
    local.points.allocate(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i) {
      auto pt = get_point(kept[i]);
      local.points[i] = tf::point<Int, 2>{pt[axes.first], pt[axes.second]};
    }

    local.fs.build(local.base_loop,
                   tf::make_edges(tf::make_blocked_range<2>(local.edges)),
                   tf::make_points(local.points));

    auto write_walk = [&](const auto &walk, tf::buffer<Index> &offsets,
                          tf::buffer<vertex_t> &vertices) {
      offsets.push_back(static_cast<Index>(vertices.size()));
      for (auto id : walk)
        vertices.push_back(local.ihm.kept_ids()[std::size_t(id)]);
    };

    if (local.fs.faces().size() == 0) {
      write_walk(local.base_loop, local.offsets, local.vertices);
      return 1;
    }

    const bool with_holes = local.fs.holes().size() != 0;
    Index k = 0;
    for (const auto &face : local.fs.faces()) {
      write_walk(face, local.offsets, local.vertices);
      if (with_holes) {
        const Index this_loop = static_cast<Index>(local.offsets.size()) - 1;
        for (auto h : local.fs.holes_for_faces()[std::size_t(k)]) {
          local.lh_entries.push_back(
              {this_loop, static_cast<Index>(local.hole_offsets.size())});
          write_walk(local.fs.holes()[h], local.hole_offsets,
                     local.hole_vertices);
        }
      }
      ++k;
    }
    return static_cast<Index>(local.fs.faces().size());
  }

  /// Group the consumed faces' descriptors per tag, mirroring the loop
  /// grouping: `_deleted` holds object ids, `_deleted_offsets` the
  /// per-tag blocks.
  auto build_deleted(tf::buffer<desc_t> &deleted_descs, std::size_t n_tags) {
    std::sort(deleted_descs.begin(), deleted_descs.end(),
              [](const desc_t &a, const desc_t &b) {
                if (a.tag != b.tag)
                  return a.tag < b.tag;
                return a.object < b.object;
              });
    _deleted.allocate(deleted_descs.size());
    for (std::size_t i = 0; i < deleted_descs.size(); ++i)
      _deleted[i] = deleted_descs[i].object;
    _deleted_offsets.allocate(n_tags + 1);
    _deleted_offsets[0] = 0;
    Index g = 0;
    for (std::size_t t = 0; t < n_tags; ++t) {
      while (g < static_cast<Index>(deleted_descs.size()) &&
             deleted_descs[g].tag == static_cast<Index>(t))
        ++g;
      _deleted_offsets[t + 1] = g;
    }
  }

  auto build_tag_offsets(std::size_t n_tags) {
    auto n = _descriptors.size();
    if (n == 0) {
      _tag_offsets.allocate(n_tags + 1);
      std::fill(_tag_offsets.begin(), _tag_offsets.end(), Index(0));
      return;
    }
    _tag_offsets.allocate(n_tags + 1);
    _tag_offsets[0] = 0;
    Index g = 0;
    for (std::size_t t = 0; t < n_tags; ++t) {
      while (g < static_cast<Index>(n) &&
             _descriptors[g].tag == static_cast<Index>(t))
        ++g;
      _tag_offsets[t + 1] = g;
    }
  }

  tf::offset_block_buffer<Index, vertex_t> _loops;
  tf::offset_block_buffer<Index, vertex_t> _holes;
  tf::offset_block_buffer<Index, Index> _loop_hole_blocks;
  tf::buffer<Index> _block_loops;      // owning loop per block
  tf::buffer<Index> _loop_hole_index;  // per loop: block id or -1
  tf::buffer<desc_t> _descriptors;
  tf::buffer<Index> _tag_offsets;
  tf::buffer<Index> _deleted;
  tf::buffer<Index> _deleted_offsets;
};

} // namespace tf
