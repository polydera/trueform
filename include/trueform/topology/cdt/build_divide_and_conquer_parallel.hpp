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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./atomic_half_edge_arena.hpp"
#include "./build_divide_and_conquer.hpp"
#include "./delaunay_execution_tuning.hpp"
#include "./delaunay_partition_node.hpp"
#include "./plan_delaunay_partitions.hpp"
#include "./sort_parallel_fallback_leaves.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/task_arena.h>
#include <utility>

namespace tf::topology::cdt {

/// Build independent Morton subtrees through local cursors into one flat edge
/// arena, then join the small residual frontier with the same serial merge
/// operation. Arena exhaustion safely falls back to the growable serial path.
template <typename Index, typename Points, typename Edges, typename KeyBuffer,
          typename Orient, typename Incircle>
auto build_divide_and_conquer_parallel(
    Points &points, Edges &edges, const KeyBuffer &keys, Index none,
    Orient orient, Incircle incircle, std::size_t darts_per_site,
    tf::buffer<delaunay_partition_node<Index>> &nodes,
    tf::buffer<std::size_t> &leaves) -> delaunay_build_result<Index> {
  const std::size_t concurrency = std::max(
      std::size_t(1), std::size_t(tbb::this_task_arena::max_concurrency()));
  if (points.size() < delaunay_execution_tuning::parallel_topology_sites ||
      concurrency < 2) {
    sort_parallel_fallback_leaves(points, keys);
    return build_divide_and_conquer<Index>(
        points, edges, keys, none, std::move(orient), std::move(incircle));
  }

  const std::size_t target_tasks =
      std::max(std::size_t(2),
               concurrency * delaunay_execution_tuning::tasks_per_worker);
  const std::size_t target_size =
      std::max(delaunay_execution_tuning::parallel_task_sites,
               (points.size() + target_tasks - 1) / target_tasks);

  nodes.clear();
  leaves.clear();
  nodes.reserve(target_tasks * 2);
  leaves.reserve(target_tasks);
  plan_delaunay_partitions<Index>(keys, target_size, nodes, leaves);
  if (leaves.size() < 2) {
    sort_parallel_fallback_leaves(points, keys);
    return build_divide_and_conquer<Index>(
        points, edges, keys, none, std::move(orient), std::move(incircle));
  }

  atomic_half_edge_arena<Edges> arena(edges, points.size() * darts_per_site);
  try {
    tf::parallel_for_each(
        tf::make_sequence_range(leaves.size()), [&](std::size_t leaf) {
          auto &node = nodes[leaves[leaf]];
          auto cursor = arena.make_cursor();
          delaunay_divide_and_conquer<
              Index, Points, typename atomic_half_edge_arena<Edges>::cursor,
              KeyBuffer, Orient, Incircle, true>
              builder(points, cursor, keys, none, orient, incircle);
          node.frontier = builder.build_partition(node.first, node.last);
          cursor.finish(none);
        });
  } catch (const typename atomic_half_edge_arena<Edges>::capacity_exceeded &) {
    edges.clear();
    sort_parallel_fallback_leaves(points, keys);
    return build_divide_and_conquer<Index>(
        points, edges, keys, none, std::move(orient), std::move(incircle));
  }
  arena.commit();

  delaunay_divide_and_conquer<Index, Points, Edges, KeyBuffer, Orient, Incircle,
                              false>
      merger(points, edges, keys, none, std::move(orient), std::move(incircle));
  for (std::size_t i = nodes.size(); i-- > 0;) {
    auto &node = nodes[i];
    if (node.is_terminal())
      continue;
    node.frontier =
        merger.join_partitions(nodes[node.children[0]].frontier,
                               nodes[node.children[1]].frontier, node.axis);
  }
  return {nodes[0].frontier[delaunay_axis::x].minimum ^ Index(1)};
}

} // namespace tf::topology::cdt
