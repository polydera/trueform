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
#include "../../core/coordinate_type.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/points.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../policy/vertex_link.hpp"
#include "../vertex_link_like.hpp"

namespace tf::topology {

/// @ingroup topology_connectivity
/// @brief Reusable neighborhood traversal state.
///
/// Holds BFS state (visited set, queue) that can be reused across calls.
/// Use for inline neighborhood iteration without allocating intermediate
/// buffers.
///
/// @tparam Index The vertex index type.
template <typename Index> struct neighborhood_applier {
  tf::hash_set<Index> visited;
  tf::buffer<Index> queue;

  /// Apply function f to each neighbor of seed within radius.
  /// @param vlink Vertex connectivity.
  /// @param seed The seed vertex.
  /// @param distance2_f Squared distance function (seed, neighbor) -> RealT.
  /// @param radius Maximum distance (will be squared internally).
  /// @param inclusive If true, apply f to seed as well.
  /// @param f Function to apply to each neighbor.
  template <typename VertexLinkPolicy, typename Distance2Func, typename RealT,
            typename F>
  void operator()(const tf::vertex_link_like<VertexLinkPolicy> &vlink,
                  Index seed, Distance2Func distance2_f, RealT radius, F &&f,
                  bool inclusive) {
    const RealT radius2 = radius * radius;

    visited.clear();
    queue.clear();

    visited.insert(seed);
    if (inclusive)
      f(seed);
    queue.push_back(seed);

    std::size_t front = 0;
    while (front < queue.size()) {
      Index vid = queue[front++];

      for (auto neighbor : vlink[vid]) {
        if (neighbor < 0 || visited.count(neighbor))
          continue;

        visited.insert(neighbor);

        if (distance2_f(seed, neighbor) <= radius2) {
          f(neighbor);
          queue.push_back(neighbor);
        }
      }
    }
  }
};

} // namespace tf::topology
