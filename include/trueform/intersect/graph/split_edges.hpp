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
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "./crossing_record.hpp"
#include "./edge.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>

namespace tf::intersect::graph {

/// Split edges at crossing points and rebuild per-face edge lists.
///
/// Single pass over per-face edge lists: each instance either passes through
/// unchanged or gets split into sub-edges. Returns same-t merge pairs
/// (multiple EE crossings at the same parametric location) for the caller
/// to combine with other merge sources. Does not remap or re-canonicalize.
template <typename Index, typename GetPoint>
auto split_edges(tf::buffer<edge_split_entry<Index>> &entries,
                 Index crossing_base,
                 const tf::buffer<tf::point<int32_t, 3>> &crossing_points,
                 tf::offset_block_buffer<Index, edge<Index>> &edge_defs,
                 tf::offset_block_buffer<Index, Index> &edges,
                 const GetPoint &get_point,
                 tf::buffer<std::array<Index, 2>> &merge_pairs) -> void {
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

  auto &edge_data = edge_defs.data_buffer();
  tf::buffer<edge<Index>> new_edge_data;

  // Snapshot the old per-face edge lists for iteration
  auto old_edges = tf::make_range(edges);

  // New offsets built during aggregation
  tf::buffer<Index> new_edges_offsets;
  new_edges_offsets.allocate(edges.size() + 1);
  new_edges_offsets[0] = 0;
  std::size_t face_i = 1;

  struct local_t {
    tf::buffer<edge<Index>> new_edges;
    tf::buffer<Index> counts;
    tf::buffer<std::array<Index, 2>> merges;
    tf::buffer<split_point<Index>> work;
  };

  auto task = [&](auto &&range, local_t &local) {
    auto get_point_f = get_point;
    local.counts.allocate(range.size());
    auto cit = local.counts.begin();

    for (const auto &face_edges : range) {
      auto old_size = local.new_edges.size();

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
        using i128 = tf::exact::int128;
        i128 dx = i128(p1[0]) - p0[0], dy = i128(p1[1]) - p0[1],
             dz = i128(p1[2]) - p0[2];

        local.work.clear();
        for (auto &entry : group) {
          auto pid = entry.point_id;
          auto q = (pid >= crossing_base) ? crossing_points[pid - crossing_base]
                                          : get_point_f(-1, pid);
          i128 t = dx * (i128(q[0]) - p0[0]) + dy * (i128(q[1]) - p0[1]) +
                   dz * (i128(q[2]) - p0[2]);
          local.work.push_back({pid, t});
        }

        // Sort by (t, point_id) — smallest point_id first for tie-breaking
        std::sort(local.work.begin(), local.work.end(),
                  [](const auto &a, const auto &b) {
                    if (a.t != b.t)
                      return a.t < b.t;
                    return a.point_id < b.point_id;
                  });

        // Detect same-t merge pairs before dedup
        for (std::size_t i = 1; i < local.work.size(); ++i) {
          if (local.work[i].t == local.work[i - 1].t &&
              local.work[i].point_id != local.work[i - 1].point_id)
            local.merges.push_back(
                {local.work[i - 1].point_id, local.work[i].point_id});
        }

        // Dedup by t (keeps smallest point_id due to sort order)
        local.work.erase_till_end(std::unique(
            local.work.begin(), local.work.end(),
            [](const auto &a, const auto &b) { return a.t == b.t; }));

        // Emit sub-edges: walk split points from e.point_0 to e.point_1
        Index prev = e.point_0;
        for (auto &s : local.work) {
          local.new_edges.push_back({e.tag, e.tag_other, e.object,
                                     e.object_other, prev, s.point_id, 0});
          prev = s.point_id;
        }
        local.new_edges.push_back(
            {e.tag, e.tag_other, e.object, e.object_other, prev, e.point_1, 0});
      }

      *cit++ = static_cast<Index>(local.new_edges.size() - old_size);
    }
  };

  auto agg = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.new_edges, new_edge_data);
    tf::core::append(local.merges, merge_pairs);
    for (auto count : local.counts) {
      new_edges_offsets[face_i] = new_edges_offsets[face_i - 1] + count;
      ++face_i;
    }
  };

  tf::blocked_reduce_sequenced_aggregate(old_edges, tf::none, local_t{}, task,
                                         agg);

  // Install new buffers, build identity _edges indices
  edge_defs.data_buffer() = std::move(new_edge_data);
  edges.offsets_buffer() = std::move(new_edges_offsets);
  auto total = edge_defs.data_buffer().size();
  edges.data_buffer().allocate(total);
  tf::parallel_iota(edges.data_buffer(), 0);
}

} // namespace tf::intersect::graph
