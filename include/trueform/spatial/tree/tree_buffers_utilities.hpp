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
#include "../../core/aabb_from.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "./tree_buffers.hpp"
#include "./build_aabb_nodes.hpp"

namespace tf::spatial {

template <typename Index, typename BV>
auto clear_tree_buffers(tree_buffers<Index, BV> &buffers) -> void {
  buffers.primitive_aabbs_buffer().clear();
  buffers.nodes_buffer().clear();
  buffers.ids_buffer().clear();
}

template <typename Index, typename BV>
auto clear_tree_buffers_core(tree_buffers_core<Index, BV> &buffers) -> void {
  buffers.nodes_buffer().clear();
  buffers.ids_buffer().clear();
}

template <typename Partitioner, typename Index, typename BV, typename Range>
auto build_tree_buffers(tree_buffers<Index, BV> &buffers,
                        const Range &primitives, tree_config config,
                        bool use_ids = false) -> void {
  if (!use_ids) {
    buffers.primitive_aabbs_buffer().allocate(primitives.size());
    tf::parallel_transform(
        primitives, buffers.primitive_aabbs_buffer(),
        [](const auto &x) { return tf::aabb_from(x); }, tf::checked);
  }
  build_tree_nodes<Partitioner>(buffers.nodes_buffer(), buffers.ids_buffer(),
                                primitives, buffers.primitive_aabbs_buffer(),
                                config, use_ids);
}

template <typename Partitioner, typename Index, typename BV, typename Range,
          typename AabbRange>
auto build_tree_nodes_with_aabbs(tree_buffers_core<Index, BV> &buffers,
                                 const Range &primitives,
                                 const AabbRange &aabbs, tree_config config,
                                 bool use_ids = false) -> void {
  build_tree_nodes<Partitioner>(buffers.nodes_buffer(), buffers.ids_buffer(),
                                primitives, aabbs, config, use_ids);
}

} // namespace tf::spatial
