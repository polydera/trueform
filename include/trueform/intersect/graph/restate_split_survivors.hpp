/*
 * Copyright (c) 2026 XLAB
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
#include "./plane_identity_collapse.hpp"
#include <array>
#include <cstddef>

namespace tf::intersect::graph {

/// The split survivors restated in a closed rewrite table that grew after
/// they were ordered: each survivor speaks its target, and a survivor that
/// became its neighbour or its root's own end leaves, as
/// @ref tf::intersect::graph::order_plane_splits states them. One serial
/// compaction over the survivors and their offsets.
template <typename Index, typename Graph, typename VertexOffsets>
auto restate_split_survivors(const Graph &g,
                             const tf::buffer<std::array<Index, 3>> &merges,
                             const VertexOffsets &vertex_offsets,
                             const tf::buffer<Index> &split_edge,
                             tf::buffer<Index> &split_offsets,
                             tf::buffer<Index> &split_data) -> void {
  std::size_t write = 0;
  for (std::size_t group = 0; group < split_edge.size(); ++group) {
    const auto begin = std::size_t(split_offsets[group]);
    const auto end = std::size_t(split_offsets[group + 1]);
    split_offsets[group] = Index(write);
    const auto ends = plane_merged_endpoints(
        merges, vertex_offsets, g.canon_group(split_edge[group])[0]);
    Index previous = Index(-1);
    for (auto at = begin; at < end; ++at) {
      auto id = split_data[at];
      const auto target = plane_merge_target(merges, Index(1), id);
      if (target != Index(-1))
        id = target;
      if (id == previous || (ends.lo_tag < 0 && id == ends.lo_id) ||
          (ends.hi_tag < 0 && id == ends.hi_id))
        continue;
      previous = id;
      split_data[write++] = id;
    }
  }
  if (split_edge.size() != 0)
    split_offsets[split_edge.size()] = Index(write);
  split_data.erase_till_end(split_data.begin() + std::ptrdiff_t(write));
}

} // namespace tf::intersect::graph
