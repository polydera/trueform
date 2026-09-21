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

#include <thread>

#include "tbb/task_group.h"

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/points.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/half_edges.hpp"
#include "./collapse_to_target.hpp"
#include "./grow_cleanup_region.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Parallel collapse edges to reach a target face count.
///
/// Partitions the mesh via BFS flood-fill and freezes the vertices at the
/// partition boundaries. The whole partition is stated before any task runs,
/// so every collapse reads a frozen structure and mutates only its own
/// interior. A sequential cleanup pass then works the boundary region
/// @ref tf::remesh::grow_cleanup_region marks.
///
/// @tparam N_BUCKETS Bucket count for the internal bucket queue.
/// @tparam Index The index type.
/// @tparam Policy The collapse policy type.
/// @tparam PointsPolicy The points policy type.
/// @param he The half-edge structure (modified in place).
/// @param points The vertex positions (modified in place).
/// @param current_faces The current number of faces.
/// @param target_faces The desired number of faces.
/// @param policy The collapse policy (must be initialized).
/// @return The number of faces removed.
template <std::size_t N_BUCKETS = 2048, typename Index, typename Policy,
          typename PointsPolicy>
auto collapse_to_target_parallel(tf::half_edges<Index> &he,
                                 tf::points<PointsPolicy> &points,
                                 Index current_faces, Index target_faces,
                                 Policy &policy) -> Index {
  if (current_faces <= target_faces)
    return 0;

  int n_parts = int(2 * std::thread::hardware_concurrency());
  Index n_faces_alloc = Index(he.face_half_edge_handles().size());
  Index n_edges = Index(he.half_edges_buffer().size() / 2);
  Index n_verts = Index(he.vertex_half_edge_handles().size());

  tf::buffer<int> face_label;
  face_label.allocate(n_faces_alloc);
  tf::parallel_fill(face_label, int(-1));

  tf::buffer<Index> bfs_queue;
  bfs_queue.allocate(n_faces_alloc);

  // frozen buffer: 0 = free, 1 = partition boundary (parallel only)
  tf::buffer<char> frozen;
  frozen.allocate(n_verts);
  tf::parallel_fill(frozen, char(0));

  using Real = tf::coordinate_type<PointsPolicy>;
  Real ratio = Real(target_faces) / Real(current_faces);

  Index faces_per_part = std::max(Index(1), current_faces / n_parts);
  Index bfs_write = 0, bfs_read = 0;
  int current_label = 0;
  Index seed = 0;

  auto enqueue_neighbors = [&](Index f) {
    auto fhe = he.face_half_edge_handles()[f];
    if (!fhe.is_valid())
      return;
    auto cur = fhe;
    for (int i = 0; i < 3; ++i) {
      auto opp = he.opposite(tf::unsafe, cur);
      if (opp.is_valid()) {
        auto nf = he.half_edge(opp).face;
        if (nf >= 0 && face_label[nf] < 0) {
          face_label[nf] = current_label;
          bfs_queue[bfs_write++] = nf;
        }
      }
      cur = he.next(tf::unsafe, cur);
    }
  };

  auto mark_frozen = [&](Index part_begin, Index part_end, int label) {
    for (Index i = part_begin; i < part_end; ++i) {
      Index f = bfs_queue[i];
      auto fhe = he.face_half_edge_handles()[f];
      if (!fhe.is_valid())
        continue;
      auto cur = fhe;
      for (int e = 0; e < 3; ++e) {
        auto opp_heh = he.opposite(tf::unsafe, cur);
        auto &opp_h = he.half_edge(opp_heh);
        if (!opp_h.is_simple() || face_label[opp_h.face] != label) {
          frozen[he.half_edge(cur).vertex] = 1;
          frozen[opp_h.vertex] = 1;
        }
        cur = he.next(tf::unsafe, cur);
      }
    }
  };

  // Collapse task for a partition
  auto dispatch_partition = [&](tbb::task_group &tg, int p, Index begin,
                                Index end) {
    tg.run([&, p, begin, end] {
      Index n_part_faces = end - begin;

      tf::buffer<Index> local_edges;
      Index part_frozen = 0;

      for (Index i = begin; i < end; ++i) {
        Index f = bfs_queue[i];
        auto fhe = he.face_half_edge_handles()[f];
        if (!fhe.is_valid())
          continue;
        auto cur = fhe;
        bool face_has_frozen = false;
        for (int e = 0; e < 3; ++e) {
          auto v0 = he.start_vertex_handle(tf::unsafe, cur).id();
          auto v1 = he.end_vertex_handle(tf::unsafe, cur).id();
          if (frozen[v0] || frozen[v1]) {
            face_has_frozen = true;
            cur = he.next(tf::unsafe, cur);
            continue;
          }

          auto opp_heh = he.opposite(tf::unsafe, cur);
          auto &opp_h = he.half_edge(opp_heh);

          if (!opp_h.is_simple()) {
            // removed / non-manifold / orientation fault — skip
          } else if (face_label[opp_h.face] != p) {
            // partition boundary — already frozen
          } else if (f < opp_h.face) {
            // interior edge, dedup: add once (smaller face id)
            local_edges.push_back(he.edge_handle(tf::unsafe, cur).id());
          }

          cur = he.next(tf::unsafe, cur);
        }
        if (face_has_frozen)
          ++part_frozen;
      }

      Index part_interior = n_part_faces - part_frozen;
      Index part_target =
          part_frozen + std::max(Index(1), Index(part_interior * ratio));

      auto frozen_range =
          tf::make_range(frozen.data(), std::size_t(frozen.size()));
      auto edge_range =
          tf::make_range(local_edges.data(), std::size_t(local_edges.size()));

      tf::remesh::collapse_to_target<N_BUCKETS>(he, points, n_part_faces,
                                                part_target, policy, edge_range,
                                                frozen_range);
    });
  };

  // === BFS + frozen marking ===

  // Block p of these offsets is the face run of partition p, and p is its
  // label.
  tf::buffer<Index> partition_offsets;
  partition_offsets.reserve(std::size_t(n_parts) + 2);
  partition_offsets.push_back(0);

  for (; current_label < n_parts; ++current_label) {
    Index part_begin = bfs_write;

    while (seed < n_faces_alloc &&
           (face_label[seed] >= 0 ||
            !he.face_half_edge_handles()[seed].is_valid()))
      ++seed;
    if (seed >= n_faces_alloc)
      break;

    face_label[seed] = current_label;
    bfs_queue[bfs_write++] = seed;

    Index part_target_bfs = bfs_read + faces_per_part;
    while (bfs_read < bfs_write && bfs_read < part_target_bfs)
      enqueue_neighbors(bfs_queue[bfs_read++]);

    bfs_read = bfs_write;

    mark_frozen(part_begin, bfs_write, current_label);
    partition_offsets.push_back(bfs_write);
  }

  int actual_parts = current_label;
  if (actual_parts == 0)
    return 0;

  // Handle remaining unlabeled faces as an extra partition
  Index extra_begin = bfs_write;
  for (Index f = 0; f < n_faces_alloc; ++f) {
    if (face_label[f] < 0 && he.face_half_edge_handles()[f].is_valid()) {
      face_label[f] = actual_parts;
      bfs_queue[bfs_write++] = f;
    }
  }
  if (bfs_write > extra_begin) {
    mark_frozen(extra_begin, bfs_write, actual_parts);
    partition_offsets.push_back(bfs_write);
  }

  // === Collapse the frozen partition's interiors ===

  tbb::task_group tg;
  for (int p = 0; p + 1 < int(partition_offsets.size()); ++p)
    dispatch_partition(tg, p, partition_offsets[p], partition_offsets[p + 1]);
  tg.wait();

  // The counts are the barrier's to state, never a partition's.
  he.recount();
  Index post_parallel_faces = he.number_of_faces();

  // === Sequential cleanup on boundary region ===

  tf::remesh::grow_cleanup_region(he, frozen);

  // 0 → frozen (interior, already handled), stamped → collapsible (partition
  // boundary region)
  auto cleanup_frozen = tf::make_mapped_range(
      tf::make_range(frozen), [](char v) -> char { return !v; });

  auto all_edges = tf::make_sequence_range(n_edges);

  tf::remesh::collapse_to_target<N_BUCKETS>(he, points, post_parallel_faces,
                                            target_faces, policy, all_edges,
                                            cleanup_frozen);

  return current_faces - he.number_of_faces();
}

} // namespace tf::remesh
