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
#include "../core/cross.hpp"
#include "../core/dot.hpp"
#include "../core/points.hpp"
#include "../core/points_buffer.hpp"
#include "../core/vector.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/zero.hpp"
#include "../topology/half_edges.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Tangential relaxation (Laplacian + tangent-plane projection).
///
/// For each interior vertex, computes the uniform Laplacian (average of
/// neighbors), then projects the displacement onto the tangent plane
/// defined by the area-weighted vertex normal.
///
/// @param he The half-edge structure.
/// @param points The vertex positions (modified in place).
/// @param old_pos Workspace buffer (reused across calls).
/// @param iterations Number of relaxation passes.
/// @param lambda Damping factor in (0, 1].
template <typename Index, typename PointsPolicy>
auto tangential_relaxation(
    const tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    int iterations = 1,
    tf::coordinate_type<PointsPolicy> lambda = 0.5) -> void {
  Index n_verts = Index(he.vertex_half_edge_handles().size());

  if (old_pos.size() < std::size_t(n_verts))
    old_pos.allocate(n_verts);

  for (int iter = 0; iter < iterations; ++iter) {
    tf::parallel_copy(points, old_pos);

    double f = double(lambda);

    tf::parallel_for_each(tf::make_sequence_range(n_verts), [&](Index v) {
      auto vhe = he.vertex_half_edge_handles()[v];
      if (!vhe.is_valid())
        return;
      if (he.is_boundary_vertex(v))
        return;

      // Walk 1-ring: accumulate neighbor barycenter and vertex normal
      tf::vector<double, tf::coordinate_dims_v<PointsPolicy>> bary = tf::zero;
      tf::vector<double, tf::coordinate_dims_v<PointsPolicy>> normal = tf::zero;
      int count = 0;

      auto cur = vhe;
      do {
        auto nv = he.end_vertex_handle(tf::unsafe, cur).id();
        bary += old_pos[nv].as_vector_view();
        ++count;

        // Accumulate face normal (area-weighted) from the face on the left
        auto h0 = cur;
        auto h1 = he.next(tf::unsafe, h0);
        auto v0 = he.start_vertex_handle(tf::unsafe, h0).id();
        auto v1 = he.end_vertex_handle(tf::unsafe, h0).id();
        auto v2 = he.end_vertex_handle(tf::unsafe, h1).id();
        auto e0 = old_pos[v1].as_vector_view() - old_pos[v0].as_vector_view();
        auto e1 = old_pos[v2].as_vector_view() - old_pos[v0].as_vector_view();
        normal += tf::cross(e0, e1);

        cur = he.rotated(cur);
        if (!cur.is_valid())
          break;
      } while (cur != vhe);

      if (count == 0)
        return;

      bary = bary / double(count);

      // Project displacement onto tangent plane
      auto pv = old_pos[v].as_vector_view();
      auto move = bary - pv;
      double n2 = tf::dot(normal, normal);
      if (n2 > 1e-30)
        move = move - (tf::dot(move, normal) / n2) * normal;

      points[v].as_vector_view() = pv + f * move;
    });
  }
}

template <typename Index, typename PointsPolicy>
auto tangential_relaxation(
    const tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    int iterations = 1,
    tf::coordinate_type<PointsPolicy> lambda = 0.5) -> void {
  tf::remesh::tangential_relaxation(he, points, old_pos, iterations, lambda);
}

} // namespace tf::remesh

namespace tf {

/// @brief Tangential relaxation (convenience, allocates workspace).
template <typename Index, typename PointsPolicy>
auto tangential_relaxation(const tf::half_edges<Index> &he,
                           tf::points<PointsPolicy> &points,
                           int iterations = 1,
                           tf::coordinate_type<PointsPolicy> lambda = 0.5)
    -> void {
  tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> old_pos;
  tf::remesh::tangential_relaxation(he, points, old_pos, iterations, lambda);
}

template <typename Index, typename PointsPolicy>
auto tangential_relaxation(const tf::half_edges<Index> &he,
                           tf::points<PointsPolicy> &&points,
                           int iterations = 1,
                           tf::coordinate_type<PointsPolicy> lambda = 0.5)
    -> void {
  tf::tangential_relaxation(he, points, iterations, lambda);
}

} // namespace tf
