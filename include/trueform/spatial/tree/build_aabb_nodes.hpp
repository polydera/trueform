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
#include "../../core/aabb.hpp"
#include "../../core/aabb_from.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/algorithm/partition_range_into_parts.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/zip.hpp"
#include "../../core/largest_axis.hpp"
#include "../../core/points.hpp"
#include "../../core/points_buffer.hpp"
#include "../tree_config.hpp"
#include "./max_nodes_in_tree.hpp"
#include "./tree_node.hpp"
namespace tf::spatial {

// Recursive helper: operates on a zip(ids, centers) range. pdq partitions
// the zip in place by center[max_axis]; ids are shuffled in lockstep. The
// outer entry point already wrote ids = 0..n-1 (or accepted user ids).
template <typename Partitioner, typename Index, typename RealT,
          std::size_t Dims, typename Range1, typename ZipRange>
auto build_tree_nodes(buffer<tree_node<Index, tf::aabb<RealT, Dims>>> &nodes,
                      const Range1 &aabbs, ZipRange &&kv,
                      Index node_id, Index offset,
                      const tf::tree_config &config) -> void {
  Index n_ids = kv.size();

  // Leaf: union the primitive AABBs of this leaf's prims, looked up via
  // the id half of the zip (std::get<0>).
  if (n_ids <= config.leaf_size) {
    nodes[node_id].bv = aabbs[std::get<0>(kv.front())];
    auto it = kv.begin();
    auto end = kv.end();
    while (++it != end)
      aabb_union_inplace(nodes[node_id].bv, aabbs[std::get<0>(*it)]);
    nodes[node_id].set_data(offset, n_ids);
    nodes[node_id].set_as_leaf();
    return;
  }

  // Inner: split-axis from the bbox of centroids. The centers sub-range
  // sits inside the points_buffer (contiguous), so tf::aabb_from hits the
  // fast blocked SIMD path via point_iterator.
  auto cbb = tf::aabb_from(tf::make_points(tf::make_range(
      std::get<1>(kv.begin().base_iter()),
      std::get<1>(kv.end().base_iter()))));

  int max_axis = tf::largest_axis(cbb.diagonal());
  nodes[node_id].axis = max_axis;

  Index first_child = config.inner_size * node_id + 1;

  Index n_children = tf::partition_range_into_parts(
      kv, config.inner_size,
      [&](auto begin, auto mid, auto end) {
        Partitioner::partition(begin, mid, end,
            [max_axis](const auto &a, const auto &b) {
              return std::get<1>(a)[max_axis] < std::get<1>(b)[max_axis];
            });
      },
      [&](auto &&range, Index this_node_id) {
        Index this_offset = Index(range.begin() - kv.begin());
        build_tree_nodes<Partitioner>(nodes, aabbs, range, this_node_id,
                                      offset + this_offset, config);
      },
      first_child);

  // Compose this node's BV from the children's already-written BVs.
  nodes[node_id].bv = nodes[first_child].bv;
  for (Index k = 1; k < n_children; ++k)
    aabb_union_inplace(nodes[node_id].bv, nodes[first_child + k].bv);

  nodes[node_id].set_data(first_child, n_children);
}

template <typename Partitioner, typename Index, typename RealT,
          std::size_t Dims, typename Range0, typename Range1>
auto build_tree_nodes(buffer<tree_node<Index, tf::aabb<RealT, Dims>>> &nodes,
                      buffer<Index> &ids, const Range0 &primitives,
                      const Range1 &aabbs, tf::tree_config config,
                      bool use_ids = false) -> void {
  (void)primitives;
  nodes.clear();
  if (!aabbs.size()) {
    ids.clear();
    return;
  }
  Index n_ids = use_ids ? Index(ids.size()) : Index(aabbs.size());
  nodes.allocate(max_nodes_in_tree(n_ids, config.inner_size, config.leaf_size));
  tf::parallel_for_each(nodes, [](auto &x) { x.set_as_empty(); }, tf::checked);
  if (!use_ids) {
    ids.allocate(aabbs.size());
    tf::parallel_iota(ids, 0);
  }

  tf::points_buffer<RealT, Dims> centers;
  centers.allocate(n_ids);
  if (!use_ids) {
    tf::parallel_for_each(tf::zip(aabbs, centers), [](auto pair) {
      auto &&[bb, c] = pair;
      for (std::size_t d = 0; d < Dims; ++d) c[d] = bb.min[d] + bb.max[d];
    }, tf::checked);
  } else {
    tf::parallel_for_each(tf::enumerate(ids), [&](auto pair) {
      auto &&[i, id] = pair;
      auto const &bb = aabbs[id];
      auto c = centers[i];
      for (std::size_t d = 0; d < Dims; ++d) c[d] = bb.min[d] + bb.max[d];
    }, tf::checked);
  }

  build_tree_nodes<Partitioner>(nodes, aabbs, tf::zip(ids, centers),
                                Index(0), Index(0), config);
}
} // namespace tf::spatial
