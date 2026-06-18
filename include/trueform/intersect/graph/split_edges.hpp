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
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/meta.hpp"
#include "./clean_loop.hpp"
#include "./crossing_record.hpp"
#include "./edge.hpp"
#include "./edges.hpp"
#include "./vertex.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace tf::intersect::graph {

/// Split edges at crossing points and rebuild per-face edge lists.
///
/// Single pass over per-face edge lists: each instance either passes through
/// unchanged or gets split into sub-edges. Returns same-t merge pairs
/// (multiple EE crossings at the same parametric location) for the caller
/// to combine with other merge sources. Does not remap or re-canonicalize.
template <typename Index, typename Int, typename GetPoint>
auto split_edges(tf::buffer<edge_split_entry<Index>> &entries,
                 Index crossing_base,
                 const tf::buffer<tf::point<Int, 3>> &crossing_points,
                 tf::buffer<edge<Index>> &edge_data,
                 tf::offset_block_buffer<Index, Index> &edges,
                 tf::offset_block_buffer<Index, vertex<Index>> &loops,
                 const GetPoint &get_point,
                 tf::buffer<std::array<Index, 2>> &merge_pairs) -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  // Sort by (edge_id, point_id), deduplicate
  tbb::parallel_sort(entries.begin(), entries.end());
  entries.erase_till_end(std::unique(entries.begin(), entries.end()));

  // Group by edge_id → split_groups + lookup
  tf::buffer<Index> split_offsets;
  split_offsets.reserve(entries.size() + 1);
  tf::compute_offsets(
      entries, std::back_inserter(split_offsets), Index(0),
      [](const auto &a, const auto &b) { return a.edge_id == b.edge_id; });
  auto split_groups =
      tf::make_offset_block_range(split_offsets, tf::make_range(entries));
  auto num_split_edges = split_offsets.size() - 1;

  tf::buffer<Index> split_group_eid;
  split_group_eid.allocate(num_split_edges);
  for (std::size_t si = 0; si < num_split_edges; ++si)
    split_group_eid[si] = split_groups.begin()[si][0].edge_id;

  auto loops_range = tf::make_range(loops);
  tf::buffer<edge<Index>> new_edge_data;
  tf::buffer<vertex<Index>> new_loop_data;

  // Snapshot the old per-face edge lists for iteration
  auto old_edges = tf::make_range(edges);

  // New offsets built during aggregation
  tf::buffer<Index> new_edges_offsets;
  new_edges_offsets.allocate(edges.size() + 1);
  new_edges_offsets[0] = 0;
  tf::buffer<Index> new_loops_offsets;
  new_loops_offsets.allocate(loops.size() + 1);
  new_loops_offsets[0] = 0;
  std::size_t face_i = 1;
  std::size_t loop_i = 1;

  struct local_t {
    tf::buffer<edge<Index>> new_edges;
    tf::buffer<Index> counts;
    tf::buffer<vertex<Index>> dirty;
    tf::buffer<Index> loop_counts;
    tf::buffer<std::array<Index, 2>> merges;
    tf::buffer<split_point<Index, Int>> work;
  };

  auto task = [&](auto &&range, local_t &local) {
    auto get_point_f = get_point;
    local.counts.allocate(range.size());
    local.loop_counts.allocate(range.size());
    auto cit = local.counts.begin();
    auto lit = local.loop_counts.begin();

    for (const auto &[face_edges, base_loop] : range) {
      auto old_size = local.new_edges.size();
      auto old_dirty = local.dirty.size();

      for (auto inst_idx : face_edges) {
        auto &e = edge_data[inst_idx];

        // Check if this canonical edge has split entries
        auto sit = std::lower_bound(split_group_eid.begin(),
                                    split_group_eid.end(), e.id);
        if (sit == split_group_eid.end() || *sit != e.id) {
          // Unsplit: emit as-is
          local.new_edges.push_back(e);
          continue;
        }

        // Has splits: compute parametric t for each split point
        auto si = static_cast<std::size_t>(sit - split_group_eid.begin());
        auto &&group = split_groups.begin()[si];

        auto p0 = get_point_f(-1, e.point_0);
        auto p1 = get_point_f(-1, e.point_1);
        T1 dx = T1(p1[0]) - p0[0], dy = T1(p1[1]) - p0[1],
           dz = T1(p1[2]) - p0[2];
        T2 t_max = T2(dx) * T2(dx) + T2(dy) * T2(dy) + T2(dz) * T2(dz);

        // Two split points whose integer coordinates differ by at most a sub-ulp
        // (squared distance <= ulp2) are the same physical point separated only
        // by construction rounding. sub_ulp() reports that; such points are fused
        // (one kept, the other routed to merge_pairs for the global remap) so no
        // degenerate sub-ulp edge survives into the rebuilt loop.
        const T2 ulp2 = T2(8);
        auto coord = [&](Index pid) {
          return (pid >= crossing_base) ? crossing_points[pid - crossing_base]
                                        : get_point_f(-1, pid);
        };
        auto sub_ulp = [&](Index a, Index b) -> bool {
          auto pa = coord(a), pb = coord(b);
          T1 ex = T1(pa[0]) - pb[0], ey = T1(pa[1]) - pb[1],
             ez = T1(pa[2]) - pb[2];
          return T2(ex) * ex + T2(ey) * ey + T2(ez) * ez <= ulp2;
        };

        local.work.clear();
        for (auto &entry : group) {
          auto pid = entry.point_id;
          auto q = (pid >= crossing_base) ? crossing_points[pid - crossing_base]
                                          : get_point_f(-1, pid);
          T2 t = T2(dx) * (T1(q[0]) - p0[0]) + T2(dy) * (T1(q[1]) - p0[1]) +
                 T2(dz) * (T1(q[2]) - p0[2]);
          local.work.push_back({pid, t});
        }

        // Sort by (t, point_id) — smallest point_id first for tie-breaking
        std::sort(local.work.begin(), local.work.end(),
                  [](const auto &a, const auto &b) {
                    if (a.t != b.t)
                      return a.t < b.t;
                    return a.point_id < b.point_id;
                  });

        // Cluster coincident split points and compact in one pass. A point joins
        // the current cluster iff it shares the anchor's exact t or lies within a
        // sub-ulp of the anchor (the cluster's kept point, smallest (t,id) by the
        // sort); the anchor is kept and the rest route to merge_pairs for the
        // global remap. Comparing against the anchor rather than the previous
        // point bounds each cluster to one sub-ulp, so consecutive kept points
        // are never sub-ulp-close and no degenerate edge survives.
        if (local.work.size()) {
          std::size_t w = 0; // write head = current cluster anchor
          for (std::size_t i = 1; i < std::size_t(local.work.size()); ++i) {
            if (local.work[i].point_id == local.work[w].point_id)
              continue; // exact duplicate of the anchor
            if (local.work[i].t == local.work[w].t ||
                sub_ulp(local.work[w].point_id, local.work[i].point_id)) {
              local.merges.push_back(
                  {local.work[w].point_id, local.work[i].point_id});
              continue; // fused into the anchor
            }
            if (++w != i)
              local.work[w] = local.work[i]; // new cluster anchor
          }
          local.work.erase_till_end(local.work.begin() + (w + 1));
        }

        // Trim split points at (or within a sub-ulp of) the edge endpoints;
        // identify them with e.point_0 / e.point_1 directly so the crossing is
        // recorded as an edge-vertex incidence rather than a tiny edge.
        auto begin = local.work.begin();
        auto end = local.work.end();
        while (begin != end &&
               (begin->t == 0 || sub_ulp(begin->point_id, e.point_0))) {
          if (begin->point_id != e.point_0)
            local.merges.push_back({e.point_0, begin->point_id});
          ++begin;
        }
        while (begin != end &&
               ((end - 1)->t == t_max ||
                sub_ulp((end - 1)->point_id, e.point_1))) {
          --end;
          if (end->point_id != e.point_1)
            local.merges.push_back({e.point_1, end->point_id});
        }

        // Emit sub-edges: walk split points from e.point_0 to e.point_1.
        // Sub-edges inherit parent's ordinal. sub_ordinal increments in
        // t-order, locally per parent split (interior parents stay -1).
        Index prev = e.point_0;
        std::int16_t sub_ord = 0;
        auto next_sub = [&]() -> std::int16_t {
          return e.ordinal < 0 ? std::int16_t(-1) : sub_ord++;
        };
        for (auto it = begin; it != end; ++it) {
          local.new_edges.push_back({e.tag, e.tag_other, e.object,
                                     e.object_other, prev, it->point_id, 0,
                                     e.ordinal, next_sub()});
          prev = it->point_id;
        }
        local.new_edges.push_back({e.tag, e.tag_other, e.object, e.object_other,
                                   prev, e.point_1, 0, e.ordinal, next_sub()});
      }

      auto part_it = std::partition(
          local.new_edges.begin() + old_size, local.new_edges.end(),
          [](const auto &edge) { return edge.ordinal == -1; });
      auto base_edges = tf::make_range(part_it, local.new_edges.end());
      build_dirty_loop<Index>(base_edges, base_loop, local.dirty);
      local.new_edges.erase_till_end(part_it);

      // Reconcile interior sub-edges against the REBUILT loop. A split point can
      // land on the boundary (it's also a split entry of a boundary edge, so
      // build_dirty_loop inserts it into the loop); the interior parent's
      // sub-edge to it then connects two now-consecutive base-loop vertices but
      // kept the parent's ordinal -1. base_loop_edge_ordinal ran at build_edges,
      // before the point existed, so the partition missed it. Re-check here and
      // drop these — their segment is already part of the loop.
      {
        auto dloop = tf::make_range(local.dirty.begin() + old_dirty,
                                    local.dirty.end());
        auto ne = std::remove_if(
            local.new_edges.begin() + old_size, local.new_edges.end(),
            [&](const auto &ed) {
              auto o = base_loop_edge_ordinal<Index>(dloop, ed.point_0,
                                                     ed.point_1);
              return o[0] != -1;
            });
        local.new_edges.erase_till_end(ne);
      }
      *cit++ = static_cast<Index>(local.new_edges.size() - old_size);
      *lit++ = static_cast<Index>(local.dirty.size() - old_dirty);
    }
  };

  auto agg = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.new_edges, new_edge_data);
    tf::core::append(local.dirty, new_loop_data);
    tf::core::append(local.merges, merge_pairs);
    for (auto count : local.counts) {
      new_edges_offsets[face_i] = new_edges_offsets[face_i - 1] + count;
      ++face_i;
    }
    for (auto count : local.loop_counts) {
      new_loops_offsets[loop_i] = new_loops_offsets[loop_i - 1] + count;
      ++loop_i;
    }
  };

  tf::blocked_reduce_sequenced_aggregate(tf::zip(old_edges, loops_range),
                                         tf::none, local_t{}, task, agg);

  // Install new buffers, build identity _edges indices
  edge_data = std::move(new_edge_data);
  edges.offsets_buffer() = std::move(new_edges_offsets);
  loops.data_buffer() = std::move(new_loop_data);
  loops.offsets_buffer() = std::move(new_loops_offsets);
  auto total = edge_data.size();
  edges.data_buffer().allocate(total);
  tf::parallel_iota(edges.data_buffer(), 0);
}

} // namespace tf::intersect::graph
