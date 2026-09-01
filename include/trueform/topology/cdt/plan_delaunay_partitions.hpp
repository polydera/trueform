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
#include "../../core/buffer.hpp"
#include "./delaunay_partition_node.hpp"
#include "./partition_morton_sites.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// Materialize the small dependency frontier used by the parallel topology
/// phase. Nodes are expanded iteratively; child indices always follow their
/// parent, so one reverse pass later joins completed children.
template <typename Index, typename KeyBuffer>
auto plan_delaunay_partitions(const KeyBuffer &keys, std::size_t target_sites,
                              tf::buffer<delaunay_partition_node<Index>> &nodes,
                              tf::buffer<std::size_t> &terminals) -> void {
  nodes.push_back({0, keys.size()});
  for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
    const std::size_t first = nodes[node_index].first;
    const std::size_t last = nodes[node_index].last;
    if (last - first <= target_sites) {
      terminals.push_back(node_index);
      continue;
    }

    const auto partition = partition_morton_sites(keys, first, last);
    if (!partition) {
      terminals.push_back(node_index);
      continue;
    }

    const std::size_t lower = nodes.size();
    nodes.push_back({first, partition->cut});
    const std::size_t upper = nodes.size();
    nodes.push_back({partition->cut, last});
    nodes[node_index].children = {lower, upper};
    nodes[node_index].axis = partition->axis;
  }
}

} // namespace tf::topology::cdt
