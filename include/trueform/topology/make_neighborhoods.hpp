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
#include "../core/coordinate_type.hpp"
#include "../core/frame_of.hpp"
#include "../core/hash_set.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/points.hpp"
#include "../core/transformed.hpp"
#include "../core/views/sequence_range.hpp"
#include "./policy/vertex_link.hpp"
#include "./vertex_link_like.hpp"

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
                  Index seed, Distance2Func distance2_f, RealT radius,
                  bool inclusive, F &&f) {
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

namespace tf {

/// @ingroup topology_connectivity
/// @brief Compute radius-based neighborhoods for all vertices.
///
/// For each vertex, computes all vertices reachable via mesh edges
/// where the squared distance from seed is within radius².
///
/// @tparam Policy The vertex link policy type.
/// @tparam Distance2Func Callable (Index seed, Index neighbor) -> RealT squared
/// distance.
/// @tparam RealT The real number type for radius.
/// @param vlink The 1-ring vertex connectivity.
/// @param distance2_f Squared distance function between vertex ids.
/// @param radius The maximum distance (will be squared internally).
/// @param inclusive If true, include the seed vertex in its own neighborhood.
/// @return Offset block buffer containing neighborhoods.
template <typename Policy, typename Distance2Func, typename RealT>
auto make_neighborhoods(const tf::vertex_link_like<Policy> &vlink,
                        Distance2Func distance2_f, RealT radius,
                        bool inclusive = false) {
  using Index = std::decay_t<decltype(vlink[0][0])>;
  const auto n_vertices = vlink.size();

  tf::offset_block_buffer<Index, Index> result;
  tf::generate_offset_blocks(
      tf::make_sequence_range(static_cast<Index>(n_vertices)), result,
      [&, applier = topology::neighborhood_applier<Index>()](
          Index seed, auto &id_buff) mutable {
        applier(vlink, seed, distance2_f, radius, inclusive,
                [&](Index neighbor) { id_buff.push_back(neighbor); });
      });

  return result;
}

/// @ingroup topology_connectivity
/// @brief Compute radius-based neighborhoods using Euclidean distance.
///
/// Convenience overload for points with vertex_link policy attached.
///
/// @tparam Policy The points policy type (must have vertex_link attached).
/// @param pts Points with vertex_link policy attached.
/// @param radius The maximum Euclidean distance.
/// @param inclusive If true, include the seed vertex in its own neighborhood.
/// @return Offset block buffer containing neighborhoods.
template <typename Policy>
auto make_neighborhoods(const tf::points<Policy> &pts,
                        tf::coordinate_type<Policy> radius,
                        bool inclusive = false) {
  static_assert(tf::has_vertex_link_policy<Policy>,
                "Points must have vertex_link policy attached");

  const auto &vlink = pts.vertex_link();
  auto frame = tf::frame_of(pts);

  return make_neighborhoods(
      vlink,
      [&](auto seed, auto neighbor) {
        auto dist_vec = tf::transformed(pts[seed] - pts[neighbor], frame);
        return dist_vec.length2();
      },
      radius, inclusive);
}

} // namespace tf
