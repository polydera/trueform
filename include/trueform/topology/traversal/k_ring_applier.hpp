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
#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../vertex_link_like.hpp"

namespace tf::topology {

/// @ingroup topology_connectivity
/// @brief Reusable k-ring traversal state.
///
/// Holds BFS state (visited set, queue) that can be reused across calls.
/// Use for inline k-ring iteration without allocating intermediate buffers.
///
/// @tparam Index The vertex index type.
template <typename Index> struct k_ring_applier {
  tf::hash_set<Index> visited;
  tf::buffer<Index> queue;

  /// Apply function f to each neighbor within k rings.
  /// @param vlink Vertex connectivity.
  /// @param seed The seed vertex.
  /// @param k Number of rings.
  /// @param inclusive If true, apply f to seed as well.
  /// @param f Function to apply to each neighbor.
  template <typename VertexLinkPolicy, typename F>
  void operator()(const tf::vertex_link_like<VertexLinkPolicy> &vlink,
                  Index seed, std::size_t k, bool inclusive, F &&f) {
    visited.clear();
    queue.clear();

    visited.insert(seed);
    if (inclusive)
      f(seed);
    queue.push_back(seed);

    std::size_t front = 0;
    std::size_t current_ring_end = queue.size();
    std::size_t ring = 0;

    while (front < queue.size() && ring < k) {
      Index vid = queue[front++];

      for (auto neighbor : vlink[vid]) {
        if (neighbor < 0 || visited.count(neighbor))
          continue;

        visited.insert(neighbor);
        f(neighbor);
        if (ring < k - 1)
          queue.push_back(neighbor);
      }

      if (front >= current_ring_end) {
        ++ring;
        current_ring_end = queue.size();
      }
    }
  }
};

} // namespace tf::topology
