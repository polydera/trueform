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

#include <algorithm>
#include <limits>

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/points.hpp"
#include "../../topology/half_edges.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Collapse edges to reach a target face count.
///
/// Greedily collapses edges in order of increasing error until the
/// target face count is reached. Uses a bucket queue for O(1) minimum
/// extraction. Only considers edges from the provided range, and
/// skips edges incident to frozen vertices.
///
/// @tparam Index The index type.
/// @tparam Policy The collapse policy type.
/// @tparam PointsPolicy The points policy type.
/// @tparam EdgeRange A range of edge IDs (Index values).
/// @param he The half-edge structure (modified in place).
/// @param points The vertex positions (modified in place).
/// @param current_faces The current number of faces in the working region.
/// @param target_faces The desired number of faces.
/// @param policy The collapse policy.
/// @param edge_ids The range of edge IDs to consider.
/// @param frozen Per-vertex protection mask. Empty means no protection.
/// @return The number of edges collapsed.
template <std::size_t N_BUCKETS = 2048, typename Index, typename Policy, typename PointsPolicy,
          typename EdgeRange, typename FrozenRange>
auto collapse_to_target(tf::half_edges<Index> &he,
                        tf::points<PointsPolicy> &points,
                        Index current_faces, Index target_faces,
                        Policy &policy, const EdgeRange &edge_ids,
                        const FrozenRange &frozen) -> Index {
  using Real = tf::coordinate_type<PointsPolicy>;
  constexpr auto max_score = std::numeric_limits<Real>::max();

  auto is_frozen = [&](Index v) {
    return frozen.size() > 0 && frozen[v];
  };

  Index remaining = current_faces - target_faces;
  if (remaining <= 0)
    return 0;

  Index n_total_edges = Index(he.half_edges_buffer().size() / 2);

  // Score buffer covers all edges; only our edges get real scores
  tf::buffer<Real> scores;
  scores.allocate(n_total_edges);
  tf::parallel_fill(scores, max_score);

  bool preserve_boundary = policy.preserve_boundary();

  tf::parallel_for_each(edge_ids, [&](Index eid) {
    auto eh = tf::edge_handle<Index>{eid};
    auto heh = policy.half_edge_to_collapse(he, eh);
    if (he.half_edge(heh).is_removed())
      return;
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    if (is_frozen(v0) || is_frozen(v1))
      return;
    if (he.is_boundary_vertex(v0) && he.is_boundary_vertex(v1) &&
        (preserve_boundary || !he.is_boundary(tf::unsafe, eh)))
      return;
    scores[eid] = policy.collapse_error(he, points, heh);
  });

  Real global_max = 0;
  for (auto eid : edge_ids)
    if (scores[eid] < max_score && scores[eid] > global_max)
      global_max = scores[eid];

  Real bucket_width = (global_max * Real(1.001)) / N_BUCKETS;
  if (bucket_width <= 0)
    bucket_width = 1;
  Real inv_bucket_width = Real(1) / bucket_width;

  constexpr Index n_buckets = Index(N_BUCKETS);

  auto to_bucket = [&](Real score) -> Index {
    return Index(std::min(std::size_t(score * inv_bucket_width),
                          N_BUCKETS - 1));
  };

  tf::buffer<Index> buckets[N_BUCKETS];
  Index bucket_head[N_BUCKETS] = {};

  for (auto eid : edge_ids) {
    if (scores[eid] < max_score)
      buckets[to_bucket(scores[eid])].push_back(eid);
  }

  Index min_bucket = 0;
  tf::buffer<Index> ring;
  Index n_collapsed = 0;

  while (remaining > 0) {
    while (min_bucket < n_buckets &&
           bucket_head[min_bucket] >= Index(buckets[min_bucket].size()))
      ++min_bucket;
    if (min_bucket >= n_buckets)
      break;

    auto eid = buckets[min_bucket][bucket_head[min_bucket]++];

    auto eh = tf::edge_handle<Index>{eid};
    auto heh0 = policy.half_edge_to_collapse(he, eh);
    if (he.half_edge(heh0).is_removed()) {
      continue;
    }

    if (scores[eid] >= max_score) {

      continue;
    }

    Index expected_b = to_bucket(scores[eid]);
    if (expected_b != min_bucket) {

      continue;
    }

    auto v0 = he.start_vertex_handle(tf::unsafe, heh0).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh0).id();

    if (is_frozen(v0) || is_frozen(v1)) {
      scores[eid] = max_score;

      continue;
    }

    if (he.is_boundary_vertex(v0) && he.is_boundary_vertex(v1) &&
        (preserve_boundary || !he.is_boundary(tf::unsafe, eh))) {
      scores[eid] = max_score;

      continue;
    }

    if (!he.is_collapse_ok(eh, ring)) {

      continue;
    }

    auto pt = policy.collapsed_point(he, points, heh0);
    if (!policy.is_collapse_allowed(he, points, heh0, pt)) {

      continue;
    }

    auto faces_removed = he.collapse(heh0);
    remaining -= faces_removed;
    policy.commit_collapse(v0, v1, pt, points);
    ++n_collapsed;

    // Rescore ring neighbors of surviving vertex
    auto vhe = he.vertex_half_edge_handles()[v0];
    if (!vhe.is_valid())
      continue;
    auto cur = vhe;
    do {
      auto neid = he.edge_handle(tf::unsafe, cur).id();
      auto neh = tf::edge_handle<Index>{neid};
      auto nheh = policy.half_edge_to_collapse(he, neh);
      if (!he.half_edge(nheh).is_removed()) {
        auto nv0 = he.start_vertex_handle(tf::unsafe, nheh).id();
        auto nv1 = he.end_vertex_handle(tf::unsafe, nheh).id();
        if (is_frozen(nv0) || is_frozen(nv1)) {
          scores[neid] = max_score;
        } else if (he.is_boundary_vertex(nv0) &&
                   he.is_boundary_vertex(nv1) &&
                   (preserve_boundary || !he.is_boundary(tf::unsafe, neh))) {
          scores[neid] = max_score;
        } else {
          auto new_score = policy.collapse_error(he, points, nheh);
          scores[neid] = new_score;
          Index b = to_bucket(new_score);
          buckets[b].push_back(neid);
          if (b < min_bucket)
            min_bucket = b;
        }
      } else {
        scores[neid] = max_score;
      }
      cur = he.rotated(cur);
      if (!cur.is_valid())
        break;
    } while (cur != vhe);
  }

  return n_collapsed;
}

/// @brief Overload without a frozen range or edge subset.
template <std::size_t N_BUCKETS = 2048, typename Index, typename Policy,
          typename PointsPolicy>
auto collapse_to_target(tf::half_edges<Index> &he,
                        tf::points<PointsPolicy> &points,
                        Index current_faces, Index target_faces,
                        Policy &policy) -> Index {
  Index n_edges = Index(he.half_edges_buffer().size() / 2);
  return tf::remesh::collapse_to_target<N_BUCKETS>(
      he, points, current_faces, target_faces, policy,
      tf::make_sequence_range(n_edges),
      tf::make_range(static_cast<const char *>(nullptr), std::size_t(0)));
}

} // namespace tf::remesh
