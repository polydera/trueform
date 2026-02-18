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

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/points.hpp"
#include "../core/points_buffer.hpp"
#include "../core/vector.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/zero.hpp"
#include "../topology/half_edges.hpp"
#include "./equalize/vertex.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Equalize triangle areas around each vertex.
///
/// Iteratively moves interior vertices toward positions that minimize
/// the sum of squared triangle areas in their 1-ring. After all
/// iterations, vertices with exactly 3 neighbors are centered among
/// them.
///
/// @param he The half-edge structure.
/// @param points The vertex positions (modified in place).
/// @param old_pos Workspace buffer for double-precision position snapshot
///   (reused across calls to avoid reallocation).
/// @param iterations Number of relaxation passes.
/// @param lambda Damping factor in (0, 1] controlling step size.
/// @param in_tangent_plane If true, constrains movement to the tangent
///   plane to prevent surface shrinkage.
template <typename Index, typename PointsPolicy>
auto equalize_areas(
    const tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    Index iterations = 1, tf::coordinate_type<PointsPolicy> lambda = 0.5,
    bool in_tangent_plane = true) -> void {
  using Real = tf::coordinate_type<PointsPolicy>;
  Index n_verts = Index(he.vertex_half_edge_handles().size());

  if (old_pos.size() < std::size_t(n_verts))
    old_pos.allocate(n_verts);

  for (Index iter = 0; iter < iterations; ++iter) {
    tf::parallel_copy(points, old_pos);

    double f = double(lambda);
    // Compute and apply new positions for interior vertices
    tf::parallel_for_each(tf::make_sequence_range(n_verts), [&](Index v) {
      auto vhe = he.vertex_half_edge_handles()[v];
      if (!vhe.is_valid())
        return;
      if (he.is_boundary_vertex(v))
        return;
      tf::point<double, tf::coordinate_dims_v<PointsPolicy>> target;
      if (!tf::remesh::equalize_areas_vertex(he, old_pos.data_buffer().data(),
                                             v, target.data(),
                                             in_tangent_plane))
        return;
      auto pv = old_pos[v];
      points[v] = pv + f * (target - pv);
    });
  }

  // Smooth valence-3 vertices: center among their 3 neighbors.
  // Two valence-3 vertices cannot be neighbors in a normal mesh,
  // so this is safe to run in parallel without double-buffering.
  tf::parallel_for_each(tf::make_sequence_range(n_verts), [&](Index v) {
    auto vhe = he.vertex_half_edge_handles()[v];
    if (!vhe.is_valid())
      return;
    if (he.is_boundary_vertex(v))
      return;
    tf::vector<double, tf::coordinate_dims_v<PointsPolicy>> sum = tf::zero;
    int count = 0;
    auto cur = vhe;
    do {
      auto nv = he.end_vertex_handle(tf::unsafe, cur).id();
      sum += points[nv].as_vector_view();
      ++count;
      cur = he.rotated(cur);
      if (!cur.is_valid())
        break;
    } while (cur != vhe);
    if (count == 3)
      points[v].as_vector_view() = sum / 3.0;
  });
}

template <typename Index, typename PointsPolicy>
auto equalize_areas(
    const tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    Index iterations = 1, tf::coordinate_type<PointsPolicy> lambda = 0.5,
    bool in_tangent_plane = true) -> void {
  tf::remesh::equalize_areas(he, points, old_pos, iterations, lambda,
                             in_tangent_plane);
}

} // namespace tf::remesh

namespace tf {

/// @ingroup remesh
/// @brief Equalize triangle areas around each vertex.
///
/// Convenience overload that allocates the workspace internally.
template <typename Index, typename PointsPolicy>
auto equalize_areas(const tf::half_edges<Index> &he,
                    tf::points<PointsPolicy> &points, Index iterations = 1,
                    tf::coordinate_type<PointsPolicy> lambda = 0.5,
                    bool in_tangent_plane = true) -> void {
  tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> old_pos;
  tf::remesh::equalize_areas(he, points, old_pos, iterations, lambda,
                             in_tangent_plane);
}

template <typename Index, typename PointsPolicy>
auto equalize_areas(const tf::half_edges<Index> &he,
                    tf::points<PointsPolicy> &&points, Index iterations = 1,
                    tf::coordinate_type<PointsPolicy> lambda = 0.5,
                    bool in_tangent_plane = true) -> void {
  tf::equalize_areas(he, points, iterations, lambda, in_tangent_plane);
}

} // namespace tf
