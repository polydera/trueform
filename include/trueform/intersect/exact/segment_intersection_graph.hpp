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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../exact/int128.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../intersections_within_segments.hpp"
#include <algorithm>

namespace tf::intersect {

template <typename Index, typename RealType, std::size_t Dims>
class segment_intersection_graph {
private:
  using vtx = tf::intersect::graph::vertex<Index>;
  using src = tf::intersect::graph::vertex_source;
  using edge_t = std::array<vtx, 2>;
  using merge_t = std::array<Index, 2>;
  using i128 = tf::exact::int128;

  struct work_node {
    Index point_id;
    i128 t;
    tf::topo_id<Index> sub_id;
  };

  struct local_t {
    tf::buffer<Index> sizes;
    tf::buffer<edge_t> data;
    tf::buffer<merge_t> merges;
    tf::buffer<work_node> work;
    tf::buffer<Index> origins;
  };

public:
  auto points() const { return tf::make_range(_points); }
  auto point_remap() const { return tf::make_range(_point_remap); }
  auto origin_edges() const { return tf::make_range(_origin_edges); }
  auto sub_edges() const { return tf::make_range(_sub_edges); }
  auto sub_edges_buffer() const
      -> const tf::offset_block_buffer<Index, edge_t> & {
    return _sub_edges;
  }

  auto flat_sub_edges() const {
    return tf::make_range(_sub_edges.data_buffer());
  }

  template <typename Policy>
  auto build(const tf::intersections_within_segments<Index, RealType, Dims> &si,
             const tf::segments<Policy> &segments) {
    auto groups = si.intersections();
    auto seg_edges = segments.edges();
    auto seg_points = segments.points();
    auto n_groups = groups.size();

    _points.allocate(si.intersection_points().size());
    tf::parallel_copy(si.intersection_points(), _points);

    _sub_edges.offsets_buffer().allocate(n_groups + 1);
    _sub_edges.offsets_buffer()[0] = 0;
    std::size_t group_i = 1;

    auto &conv = si.converter();

    auto task = [&](auto &&range, local_t &local) {
      local.sizes.allocate(range.size());
      auto sit = local.sizes.begin();
      for (const auto &group : range) {
        auto old_size = local.data.size();
        if (group.size() > 0) {
          local.origins.push_back(group.front().object);
          process_group(group, seg_edges, seg_points, conv, local);
        }
        *sit++ = static_cast<Index>(local.data.size() - old_size);
      }
    };

    tf::buffer<merge_t> all_merges;

    auto agg = [&](const local_t &local, const tf::none_t &) {
      tf::core::append(local.data, _sub_edges.data_buffer());
      tf::core::append(local.merges, all_merges);
      tf::core::append(local.origins, _origin_edges);
      for (auto sz : local.sizes) {
        _sub_edges.offsets_buffer()[group_i] =
            _sub_edges.offsets_buffer()[group_i - 1] + sz;
        ++group_i;
      }
    };

    tf::blocked_reduce_sequenced_aggregate(groups, tf::none, local_t{}, task,
                                           agg);

    apply_merges(all_merges);
  }

private:
  template <typename Group, typename Edges, typename Points, typename Conv>
  auto process_group(const Group &group, const Edges &seg_edges,
                     const Points &seg_points, const Conv &conv,
                     local_t &local) {
    auto edge_id = group.front().object;
    Index v0 = seg_edges[edge_id][0];
    Index v1 = seg_edges[edge_id][1];

    auto p0 = conv(seg_points[v0]);
    auto p1 = conv(seg_points[v1]);
    i128 dx = i128(p1[0]) - p0[0];
    i128 dy = i128(p1[1]) - p0[1];
    i128 dz = i128(0);
    if constexpr (Dims == 3)
      dz = i128(p1[2]) - p0[2];

    // 1. Fill work buffer
    local.work.clear();
    for (const auto &rec : group) {
      if (rec.target.label == tf::topo_type::vertex) {
        local.work.push_back({rec.id, i128(rec.target.id), rec.target});
      } else {
        auto q = _points[rec.id];
        i128 t = dx * (i128(q[0]) - p0[0]) + dy * (i128(q[1]) - p0[1]);
        if constexpr (Dims == 3)
          t += dz * (i128(q[2]) - p0[2]);
        local.work.push_back({rec.id, t, rec.target});
      }
    }

    // 2. Partition: [v0 | edge | v1]
    auto v0_end =
        std::partition(local.work.begin(), local.work.end(), [](const auto &w) {
          return w.sub_id ==
                 tf::topo_id<Index>{Index(0), tf::topo_type::vertex};
        });
    auto v1_start = std::partition(v0_end, local.work.end(), [](const auto &w) {
      return w.sub_id != tf::topo_id<Index>{Index(1), tf::topo_type::vertex};
    });

    bool has_v0 = v0_end != local.work.begin();
    bool has_v1 = v1_start != local.work.end();

    // 3. Sort edge entries by (t, point_id)
    std::sort(v0_end, v1_start, [](const auto &a, const auto &b) {
      return std::make_pair(a.t, a.point_id) < std::make_pair(b.t, b.point_id);
    });

    // 4. Unique on whole buffer by (sub_id, point_id)
    auto w_end = std::unique(
        local.work.begin(), local.work.end(), [](const auto &a, const auto &b) {
          return a.sub_id == b.sub_id && a.point_id == b.point_id;
        });

    // 5. Same-t merge pairs
    for (auto it = local.work.begin() + 1; it < w_end; ++it) {
      auto prev = it - 1;
      if (it->t == prev->t && it->point_id != prev->point_id)
        local.merges.push_back({std::min(prev->point_id, it->point_id),
                                std::max(prev->point_id, it->point_id)});
    }

    // 6. Unique whole buffer by t
    w_end =
        std::unique(local.work.begin(), w_end,
                    [](const auto &a, const auto &b) { return a.t == b.t; });

    // 7. Emit sub-edges
    auto emit_begin = has_v0 ? local.work.begin() + 1 : local.work.begin();
    auto emit_end = has_v1 ? w_end - 1 : w_end;

    vtx start_v =
        has_v0 ? vtx{src::created,
                     local.work.begin()->point_id,
                     {short(0), tf::topo_type::vertex}}
               : vtx{src::original, v0, {short(0), tf::topo_type::vertex}};
    vtx end_v = has_v1
                    ? vtx{src::created,
                          (w_end - 1)->point_id,
                          {short(1), tf::topo_type::vertex}}
                    : vtx{src::original, v1, {short(1), tf::topo_type::vertex}};

    auto make_vtx = [](const work_node &w) -> vtx {
      return {src::created, w.point_id, {short(w.sub_id.id), w.sub_id.label}};
    };

    vtx prev_v = start_v;
    for (auto it = emit_begin; it != emit_end; ++it) {
      auto cur = make_vtx(*it);
      local.data.push_back({prev_v, cur});
      prev_v = cur;
    }
    local.data.push_back({prev_v, end_v});
  }

  auto apply_merges(tf::buffer<merge_t> &merges) {
    if (merges.size() > 0) {
      _point_remap.allocate(_points.size());
      auto n_classes =
          tf::make_dense_equivalence_class_map(merges, _point_remap);

      tf::buffer<tf::point<int32_t, Dims>> new_pts;
      new_pts.allocate(n_classes);
      tf::parallel_copy(_points,
                        tf::make_indirect_range(_point_remap, new_pts));

      tf::parallel_for_each(_sub_edges.data_buffer(), [&](edge_t &e) {
        if (e[0].source == src::created)
          e[0].id = _point_remap[e[0].id];
        if (e[1].source == src::created)
          e[1].id = _point_remap[e[1].id];
      });

      _points = std::move(new_pts);
    } else {
      _point_remap.allocate(_points.size());
      tf::parallel_iota(_point_remap, Index(0));
    }
  }

  tf::buffer<tf::point<int32_t, Dims>> _points;
  tf::buffer<Index> _point_remap;
  tf::buffer<Index> _origin_edges;
  tf::offset_block_buffer<Index, edge_t> _sub_edges;
};

} // namespace tf::intersect
