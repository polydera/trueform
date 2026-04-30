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
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "./edge.hpp"

namespace tf::intersect::graph {

/// Drop base-loop edges (ordinal != -1) from per-face edge lists. Output
/// `edge_data` is flat (non-canonical) with identity `edges` indices;
/// caller invokes `canonicalize_edges` to (re)build group structure.
template <typename Index>
auto strip_base_loop_edges(
    tf::buffer<edge<Index>> &edge_data,
    tf::offset_block_buffer<Index, Index> &edges) -> void {
  auto old_edges = tf::make_range(edges);

  tf::buffer<edge<Index>> new_edge_data;
  tf::buffer<Index> new_edges_offsets;
  new_edges_offsets.allocate(edges.size() + 1);
  new_edges_offsets[0] = 0;
  std::size_t face_i = 1;

  struct local_t {
    tf::buffer<edge<Index>> new_edges;
    tf::buffer<Index> counts;
  };

  auto task = [&](auto &&range, local_t &local) {
    local.counts.allocate(range.size());
    auto cit = local.counts.begin();
    for (const auto &face_edges : range) {
      auto old_size = local.new_edges.size();
      for (auto inst_idx : face_edges) {
        auto &e = edge_data[inst_idx];
        if (e.ordinal == -1)
          local.new_edges.push_back(e);
      }
      *cit++ = static_cast<Index>(local.new_edges.size() - old_size);
    }
  };

  auto agg = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.new_edges, new_edge_data);
    for (auto count : local.counts) {
      new_edges_offsets[face_i] = new_edges_offsets[face_i - 1] + count;
      ++face_i;
    }
  };

  tf::blocked_reduce_sequenced_aggregate(old_edges, tf::none, local_t{}, task,
                                         agg);

  edge_data = std::move(new_edge_data);
  edges.offsets_buffer() = std::move(new_edges_offsets);
  auto total = edge_data.size();
  edges.data_buffer().allocate(total);
  tf::parallel_iota(edges.data_buffer(), 0);
}

} // namespace tf::intersect::graph
