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

#include <limits>

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/cross.hpp"
#include "../core/dot.hpp"
#include "../core/none.hpp"
#include "../core/points.hpp"
#include "../core/points_buffer.hpp"
#include "../core/vector.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/zero.hpp"
#include "../core/sqrt.hpp"
#include "../topology/half_edges.hpp"
#include "./feature_mask.hpp"

#include <array>
#include <type_traits>

namespace tf::remesh {

namespace detail {

/// Shared relaxation core: `mask` is a feature_mask (crease/corner vertices
/// stay fixed) or tf::none.
template <typename Index, typename PointsPolicy, typename MaskOrNone>
auto tangential_relaxation_impl(
    const tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    const MaskOrNone &mask, int iterations,
    tf::coordinate_type<PointsPolicy> lambda,
    tf::coordinate_type<PointsPolicy> max_deviation) -> void {
  Index n_verts = Index(he.vertex_half_edge_handles().size());

  if (old_pos.size() < std::size_t(n_verts))
    old_pos.allocate(n_verts);

  for (int iter = 0; iter < iterations; ++iter) {
    tf::parallel_copy(points, old_pos);

    double f = double(lambda);

    tf::parallel_for_each(tf::make_sequence_range(n_verts), [&](Index v) {
      constexpr bool HasMask = !std::is_same_v<MaskOrNone, tf::none_t>;
      auto vhe = he.vertex_half_edge_handles()[v];
      if (!vhe.is_valid())
        return;
      if (he.is_boundary_vertex(v))
        return;
      // A non-manifold vertex belongs to several fans; the 1-ring walk
      // (he.rotated) only spins through one of them, so its Laplacian
      // barycenter is computed from a partial neighborhood. There is no
      // well-defined one-ring to smooth over — leave it fixed.
      if (he.is_non_manifold_vertex(v))
        return;
      if constexpr (HasMask)
        if (mask.vertex_type(v) != vertex_feature_type::regular)
          return;

      // Walk 1-ring: accumulate neighbor barycenter and vertex normal
      tf::vector<double, tf::coordinate_dims_v<PointsPolicy>> bary = tf::zero;
      tf::vector<double, tf::coordinate_dims_v<PointsPolicy>> normal =
          tf::zero;
      std::array<tf::vector<double, tf::coordinate_dims_v<PointsPolicy>>, 64>
          ring;
      int count = 0;

      auto cur = vhe;
      do {
        auto nv = he.end_vertex_handle(tf::unsafe, cur).id();
        if (max_deviation > 0) {
          if (count == int(ring.size()))
            return;
          ring[count] =
              old_pos[nv].as_vector_view() - old_pos[v].as_vector_view();
        }
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
        // invalid handle mid-ring = broken fan: never relax such a vertex
        if (!cur.is_valid())
          return;
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
      move = f * move;

      // Budgeted mode: bound the move instead of freezing the vertex.
      // Sliding t toward a neighbor at height h over edge L displaces the
      // surface by ~t*h/L, so t <= max_deviation*L/h; and t <= 0.45*L_min so
      // simultaneously moving neighbors can never cross their shared link.
      if (max_deviation > 0) {
        if (n2 <= 1e-30)
          return;
        double inv_n = 1.0 / tf::sqrt(n2);
        double h = 0;
        double min_e2 = std::numeric_limits<double>::max();
        for (int i = 0; i < count; ++i) {
          h = std::max(h, std::abs(tf::dot(ring[i], normal)) * inv_n);
          min_e2 = std::min(min_e2, tf::dot(ring[i], ring[i]));
        }
        double min_e = tf::sqrt(min_e2);
        double cap = 0.45 * min_e;
        if (h > double(max_deviation))
          cap = std::min(cap, double(max_deviation) * min_e / h);
        double m2 = tf::dot(move, move);
        if (m2 > cap * cap)
          move = move * (cap / tf::sqrt(m2));
      }

      points[v].as_vector_view() = pv + move;
    });
  }
}

} // namespace detail

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
/// @param max_deviation When > 0, the move length is bounded so the surface
///        stays within this budget (see improve_config).
template <typename Index, typename PointsPolicy>
auto tangential_relaxation(
    const tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    int iterations = 1, tf::coordinate_type<PointsPolicy> lambda = 0.5,
    tf::coordinate_type<PointsPolicy> max_deviation = 0) -> void {
  detail::tangential_relaxation_impl(he, points, old_pos, tf::none,
                                     iterations, lambda, max_deviation);
}

template <typename Index, typename PointsPolicy>
auto tangential_relaxation(
    const tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    int iterations = 1, tf::coordinate_type<PointsPolicy> lambda = 0.5,
    tf::coordinate_type<PointsPolicy> max_deviation = 0) -> void {
  detail::tangential_relaxation_impl(he, points, old_pos, tf::none,
                                     iterations, lambda, max_deviation);
}

/// @ingroup remesh
/// @brief Tangential relaxation (feature-aware).
///
/// Same as the base overload, but skips crease and corner vertices
/// to preserve feature edges. Regular vertices relax normally.
template <typename Index, typename PointsPolicy>
auto tangential_relaxation(
    const tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    const feature_mask &mask, int iterations = 1,
    tf::coordinate_type<PointsPolicy> lambda = 0.5,
    tf::coordinate_type<PointsPolicy> max_deviation = 0) -> void {
  detail::tangential_relaxation_impl(he, points, old_pos, mask, iterations,
                                     lambda, max_deviation);
}

template <typename Index, typename PointsPolicy>
auto tangential_relaxation(
    const tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    const feature_mask &mask, int iterations = 1,
    tf::coordinate_type<PointsPolicy> lambda = 0.5,
    tf::coordinate_type<PointsPolicy> max_deviation = 0) -> void {
  detail::tangential_relaxation_impl(he, points, old_pos, mask, iterations,
                                     lambda, max_deviation);
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

/// @brief Tangential relaxation (feature-aware, allocates workspace).
template <typename Index, typename PointsPolicy>
auto tangential_relaxation(const tf::half_edges<Index> &he,
                           tf::points<PointsPolicy> &points,
                           const tf::remesh::feature_mask &mask,
                           int iterations = 1,
                           tf::coordinate_type<PointsPolicy> lambda = 0.5)
    -> void {
  tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> old_pos;
  tf::remesh::tangential_relaxation(he, points, old_pos, mask, iterations,
                                     lambda);
}

template <typename Index, typename PointsPolicy>
auto tangential_relaxation(const tf::half_edges<Index> &he,
                           tf::points<PointsPolicy> &&points,
                           const tf::remesh::feature_mask &mask,
                           int iterations = 1,
                           tf::coordinate_type<PointsPolicy> lambda = 0.5)
    -> void {
  tf::tangential_relaxation(he, points, mask, iterations, lambda);
}

} // namespace tf
