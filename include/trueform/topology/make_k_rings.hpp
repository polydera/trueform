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

#include "../core/algorithm/generate_offset_blocks.hpp"
#include "../core/buffer.hpp"
#include "../core/hash_set.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/views/sequence_range.hpp"
#include "./vertex_link_like.hpp"

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

namespace tf {

/// @ingroup topology_connectivity
/// @brief Compute k-ring neighborhoods for all vertices.
///
/// For each vertex, computes all vertices reachable within k hops
/// along mesh edges. The 1-ring is the immediate neighbors, 2-ring
/// includes neighbors of neighbors, etc.
///
/// @tparam Policy The vertex link policy type.
/// @param vlink The 1-ring vertex connectivity.
/// @param k The number of rings (hops).
/// @param inclusive If true, include the seed vertex in its own neighborhood.
/// @return Offset block buffer containing k-ring neighborhoods.
template <typename Policy>
auto make_k_rings(const tf::vertex_link_like<Policy> &vlink, std::size_t k,
                  bool inclusive = false) {
  using Index = std::decay_t<decltype(vlink[0][0])>;
  const auto n_vertices = vlink.size();

  tf::offset_block_buffer<Index, Index> result;
  tf::generate_offset_blocks(
      tf::make_sequence_range(static_cast<Index>(n_vertices)), result,
      [&, visited = tf::hash_set<Index>(),
       queue = tf::buffer<Index>()](Index seed, auto &id_buff) mutable {
        visited.clear();
        queue.clear();

        // Mark seed as visited, optionally include in output
        visited.insert(seed);
        if (inclusive)
          id_buff.push_back(seed);
        queue.push_back(seed);

        // BFS up to k rings
        std::size_t front = 0;
        std::size_t current_ring_end = queue.size();
        std::size_t ring = 0;

        while (front < queue.size() && ring < k) {
          Index vid = queue[front++];

          for (auto neighbor : vlink[vid]) {
            if (visited.count(neighbor))
              continue;

            visited.insert(neighbor);
            id_buff.push_back(neighbor);
            // Don't push to queue on last ring - we won't expand them anyway
            if (ring < k - 1)
              queue.push_back(neighbor);
          }

          // Check if we've finished current ring
          if (front >= current_ring_end) {
            ++ring;
            current_ring_end = queue.size();
          }
        }
      });

  return result;
}

} // namespace tf
