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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/algorithm/partition_range_into_parts.hpp"
#include "../../core/buffer.hpp"
#include "../../core/largest_axis.hpp"
#include "../tree_config.hpp"
#include "./max_nodes_in_tree.hpp"
#include "./tree_node.hpp"
namespace tf::spatial {

template <typename Partitioner, typename Index, typename RealT,
          std::size_t Dims, typename Range0, typename Range1, typename Range2>
auto build_tree_nodes(buffer<tree_node<Index, tf::aabb<RealT, Dims>>> &nodes,
                      const Range0 &, const Range1 &aabbs, Range2 &ids,
                      Index node_id, Index offset,
                      const tf::tree_config &config) {
  // create the bounding box
  {
    nodes[node_id].bv = aabbs[ids.front()];
    auto it = ids.begin();
    auto end = ids.end();
    while (++it != end)
      aabb_union_inplace(nodes[node_id].bv, aabbs[*it]);
  }
  Index n_ids = ids.size();
  if (n_ids <= config.leaf_size) {
    nodes[node_id].set_data(offset, ids.size());
    nodes[node_id].set_as_leaf();
    return;
  }

  int max_axis = tf::largest_axis(nodes[node_id].bv.diagonal());
  nodes[node_id].axis = max_axis;

  Index n_children = tf::partition_range_into_parts(
      ids, config.inner_size,
      [&](auto begin, auto mid, auto end) {
        Partitioner::partition(begin, mid, end, [&](auto id0, auto id1) {
          return (aabbs[id0].max[max_axis] + aabbs[id0].min[max_axis]) / 2 <
                 (aabbs[id1].max[max_axis] + aabbs[id1].min[max_axis]) / 2;
        });
      },
      [&](auto &&range, Index this_node_id) {
        Index this_offset = range.begin() - ids.begin();
        build_tree_nodes<Partitioner>(nodes, aabbs, aabbs, range, this_node_id,
                                      offset + this_offset, config);
      },
      config.inner_size * node_id + 1);

  nodes[node_id].set_data(config.inner_size * node_id + 1, n_children);
}

template <typename Partitioner, typename Index, typename RealT,
          std::size_t Dims, typename Range0, typename Range1>
auto build_tree_nodes(buffer<tree_node<Index, tf::aabb<RealT, Dims>>> &nodes,
                      buffer<Index> &ids, const Range0 &primitives,
                      const Range1 &aabbs, tf::tree_config config,
                      bool use_ids = false) {

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
  return build_tree_nodes<Partitioner>(nodes, primitives, aabbs, ids, Index(0),
                                       Index(0), config);
}
} // namespace tf::spatial
