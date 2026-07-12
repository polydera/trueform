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

#include "../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../core/buffer.hpp"
#include "../core/none.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/views/slice.hpp"
#include "../exact/projection_axes.hpp"
#include "../exact_coordinate_converter.hpp"
#include "../intersect/for_each_cut_chord.hpp"
#include "../intersect/graph/face_descriptor.hpp"
#include "../intersect/graph/intersection_graph.hpp"
#include "../intersect/graph/loop.hpp"
#include "../intersect/types/simple_intersections.hpp"
#include "./face_cutter.hpp"
#include <algorithm>

namespace tf {

/// Splits all intersected faces using intersection_graph output.
///
/// Iterates faces from the intersection graph, runs face_cutter on each,
/// and collects all subdivided loops with their descriptors.
template <typename Index, typename Int = tf::exact::int32> class face_cuts {
  using vertex_t = intersect::graph::vertex<Index>;
  using desc_t = intersect::graph::face_descriptor<Index>;

public:
  auto loops() const { return tf::make_range(_loops); }
  auto loops_buffer() const
      -> const tf::offset_block_buffer<Index, vertex_t> & {
    return _loops;
  }
  auto descriptors() const { return tf::make_range(_descriptors); }
  auto tag_offsets() const { return tf::make_range(_tag_offsets); }

  auto clear() {
    _loops.clear();
    _descriptors.clear();
    _tag_offsets.clear();
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
    auto n = descs.size();
    if (n == 0)
      return build_tag_offsets(ig.tag_offsets().size() - 1);

    struct local_t {
      tf::buffer<Index> offsets;
      tf::buffer<vertex_t> vertices;
      tf::buffer<desc_t> descs;
      tf::buffer<std::array<Index, 2>> edge_buf;
      tf::face_cutter<Index, Int> cutter;
    };

    auto task = [&](auto &&range, local_t &local) {
      auto apply_to_face_f = apply_to_face;
      auto get_mesh_point_f = get_mesh_point;
      for (const auto &[desc, loop, edges] : range) {
        if (desc.tag == Index(-1))
          continue;
        // Skip base loops with fewer than 3 distinct vertices: nothing can
        // be triangulated and emitting them would yield degenerate faces.
        if (loop.size() < 3)
          continue;
        auto get_point = [&, &desc =
                                 desc](const vertex_t &v) -> tf::point<Int, 3> {
          if (v.source == intersect::graph::vertex_source::created)
            return pts[v.id];
          return get_mesh_point_f(desc.tag, v.id);
        };
        apply_to_face_f(
            desc.tag, desc.object,
            [&, &desc = desc, &loop = loop, &edges = edges](const auto &face) {
              using source = intersect::graph::vertex_source;
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
              auto get_projected_point =
                  [&, axes](const vertex_t &v) -> tf::point<Int, 2> {
                auto pt = get_point(v);
                return {pt[axes.first], pt[axes.second]};
              };
              local.edge_buf.clear();
              for (auto &&e : edges) {
                local.edge_buf.push_back({std::min(e.point_0, e.point_1),
                                          std::max(e.point_0, e.point_1)});
              }
              std::sort(local.edge_buf.begin(), local.edge_buf.end());
              local.edge_buf.erase_till_end(
                  std::unique(local.edge_buf.begin(), local.edge_buf.end()));
              local.edge_buf.erase_till_end(
                  std::remove_if(local.edge_buf.begin(), local.edge_buf.end(),
                                 [](auto e) { return e[0] == e[1]; }));
              auto n_loops = local.cutter.build(
                  loop, tf::make_range(local.edge_buf), get_projected_point,
                  local.offsets, local.vertices);
              for (Index i = 0; i < n_loops; ++i)
                local.descs.push_back(desc);
            });
      }
    };

    auto agg = [&](const local_t &local, const tf::none_t &) {
      if (local.offsets.size() == 0)
        return;
      auto offset = static_cast<Index>(_loops.data_buffer().size());
      auto old_off = _loops.offsets_buffer().size();
      _loops.offsets_buffer().reallocate(old_off + local.offsets.size());
      for (std::size_t i = 0; i < local.offsets.size(); ++i)
        _loops.offsets_buffer()[old_off + i] = local.offsets[i] + offset;
      tf::core::append(local.vertices, _loops.data_buffer());
      tf::core::append(local.descs, _descriptors);
    };

    tf::blocked_reduce_sequenced_aggregate(tf::zip(descs, all_loops, all_edges),
                                           tf::none, local_t{}, task, agg);
    if (_loops.offsets_buffer().size())
      _loops.offsets_buffer().push_back(
          static_cast<Index>(_loops.data_buffer().size()));

    build_tag_offsets(ig.tag_offsets().size() - 1);
  }

  /// Build from simple_intersections (isocurves/isobands).
  /// Single tag; each face is split by all of its cut chords at once. The
  /// base loop mirrors intersection_graph: edge hits are inserted in
  /// parametric order, a vertex hit takes the original vertex's place.
  template <typename Policy, typename RealT, std::size_t Dims>
  auto build(const tf::polygons<Policy> &polygons,
             const tf::intersect::simple_intersections<Index, RealT, Dims> &si)
      -> void {
    clear();
    auto subranges = si.intersections();
    if (subranges.size() == 0)
      return build_tag_offsets(1);

    auto conv = tf::make_exact_coordinate_converter(polygons);
    auto ipts = si.intersection_points();
    auto pts = polygons.points();
    auto faces = polygons.faces();
    auto get_point = [&](Index tag, Index id) -> tf::point<Int, 3> {
      return tag == -1 ? conv(ipts[id]) : conv(pts[id]);
    };
    auto get_flat_id = [](const auto &) { return Index(0); };

    struct local_t {
      tf::buffer<Index> offsets;
      tf::buffer<vertex_t> vertices;
      tf::buffer<desc_t> descs;
      tf::buffer<intersect::graph::loop_node<Index, Int>> work;
      tf::buffer<vertex_t> loop;
      tf::buffer<std::array<Index, 2>> edge_buf;
      tf::face_cutter<Index, Int> cutter;
    };

    auto task = [&](auto &&range, local_t &local) {
      for (const auto &subrange : range) {
        auto object = subrange[0].object;
        auto face = faces[object];
        auto n = static_cast<Index>(face.size());

        local.loop.clear();
        intersect::graph::extract_loop(subrange, face, Index(0), get_point,
                                       get_flat_id, local.work, local.loop);

        local.edge_buf.clear();
        for (std::size_t i = 0; i < subrange.size();) {
          std::size_t j = i + 1;
          while (j < subrange.size() && subrange[j].cut == subrange[i].cut)
            ++j;
          // chords along a boundary edge cut nothing
          tf::intersect::for_each_cut_chord(
              tf::slice(subrange, i, j), n, ipts,
              [&](Index id0, Index id1, bool on_boundary) {
                if (!on_boundary)
                  local.edge_buf.push_back(
                      {std::min(id0, id1), std::max(id0, id1)});
              });
          i = j;
        }

        auto fp = polygons[object];
        auto axes = tf::exact::projection_axes(conv(fp[0]), conv(fp[1]),
                                               conv(fp[2]));
        auto get_projected_point =
            [&, axes](const vertex_t &v) -> tf::point<Int, 2> {
          auto pt = (v.source == intersect::graph::vertex_source::original)
                        ? conv(pts[v.id])
                        : conv(ipts[v.id]);
          return {pt[axes.first], pt[axes.second]};
        };

        auto n_loops = local.cutter.build(
            tf::make_range(local.loop), tf::make_range(local.edge_buf),
            get_projected_point, local.offsets, local.vertices);
        for (Index i = 0; i < n_loops; ++i)
          local.descs.push_back({0, object});
      }
    };

    auto agg = [&](const local_t &local, const tf::none_t &) {
      if (local.offsets.size() == 0)
        return;
      auto offset = static_cast<Index>(_loops.data_buffer().size());
      auto old_off = _loops.offsets_buffer().size();
      _loops.offsets_buffer().reallocate(old_off + local.offsets.size());
      for (std::size_t i = 0; i < local.offsets.size(); ++i)
        _loops.offsets_buffer()[old_off + i] = local.offsets[i] + offset;
      tf::core::append(local.vertices, _loops.data_buffer());
      tf::core::append(local.descs, _descriptors);
    };

    tf::blocked_reduce_sequenced_aggregate(subranges, tf::none, local_t{},
                                           task, agg);
    if (_loops.offsets_buffer().size())
      _loops.offsets_buffer().push_back(
          static_cast<Index>(_loops.data_buffer().size()));
    build_tag_offsets(1);
  }

private:
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
  tf::buffer<desc_t> _descriptors;
  tf::buffer<Index> _tag_offsets;
};

} // namespace tf
