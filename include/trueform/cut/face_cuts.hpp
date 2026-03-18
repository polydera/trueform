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
#include "../exact/projection_axes.hpp"
#include "../intersect/graph/face_descriptor.hpp"
#include "../intersect/graph/intersection_graph.hpp"
#include "./face_cutter.hpp"

namespace tf {

/// Splits all intersected faces using intersection_graph output.
///
/// Iterates faces from the intersection graph, runs face_cutter on each,
/// and collects all subdivided loops with their descriptors.
template <typename Index> class face_cuts {
  using vertex_t = intersect::graph::vertex<Index>;
  using desc_t = intersect::graph::face_descriptor<Index>;

public:
  auto loops() const { return tf::make_range(_loops); }
  auto descriptors() const { return tf::make_range(_descriptors); }
  auto tag_offsets() const { return tf::make_range(_tag_offsets); }

  auto clear() {
    _loops.clear();
    _descriptors.clear();
    _tag_offsets.clear();
  }

  template <typename GetFace, typename GetMeshPoint>
  auto build(const tf::intersection_graph<Index> &ig, const GetFace &get_face,
             const GetMeshPoint &get_mesh_point) -> void {
    clear();
    auto descs = ig.descriptors();
    auto all_loops = ig.loops();
    auto all_edges = ig.edges();
    auto pts = ig.points();
    auto n = descs.size();
    if (n == 0)
      return;

    struct local_t {
      tf::buffer<Index> offsets;
      tf::buffer<vertex_t> vertices;
      tf::buffer<desc_t> descs;
      tf::face_cutter<Index> cutter;
    };

    auto task = [&](auto &&range, local_t &local) {
      auto get_face_f = get_face;
      auto get_mesh_point_f = get_mesh_point;
      for (const auto &[desc, loop, edges] : range) {
        if (desc.tag == Index(-1))
          continue;
        auto get_point =
            [&, &desc = desc](const vertex_t &v) -> tf::point<int32_t, 3> {
          if (v.source == intersect::graph::vertex_source::created)
            return pts[v.id];
          return get_mesh_point_f(desc.tag, v.id);
        };
        auto face = get_face_f(desc.tag, desc.object);
        using source = intersect::graph::vertex_source;
        auto axes = tf::exact::projection_axes(
            get_point(vertex_t{source::original, Index(face[0]), 0}),
            get_point(vertex_t{source::original, Index(face[1]), 0}),
            get_point(vertex_t{source::original, Index(face[2]), 0}));
        auto get_projected_point =
            [&, axes](const vertex_t &v) -> tf::point<int32_t, 2> {
          auto pt = get_point(v);
          return {pt[axes.first], pt[axes.second]};
        };
        auto n_loops = local.cutter.build(loop, edges, get_projected_point,
                                          local.offsets, local.vertices);
        for (Index i = 0; i < n_loops; ++i)
          local.descs.push_back(desc);
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

private:
  auto build_tag_offsets(std::size_t n_tags) {
    auto n = _descriptors.size();
    if (n == 0)
      return;
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
