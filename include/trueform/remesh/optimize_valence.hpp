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

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "../core/buffer.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/cross.hpp"
#include "../core/dot.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/points.hpp"
#include "../core/vector.hpp"
#include "../topology/half_edges.hpp"
#include "./feature_mask.hpp"
#include "./valence_deviation.hpp"

namespace tf::remesh {

namespace detail {

/// Fold guard for an edge flip. The flip turns edge (va0,va1) -- shared by the
/// triangles (va0,va1,va2) and (va1,va0,vb2) -- into edge (va2,vb2), giving the
/// triangles (va2,vb2,va1) and (vb2,va2,va0). Returns false if either new
/// triangle is inverted relative to the old triangle it replaces, i.e. the
/// quad is non-convex so the flip would fold the surface. is_flip_ok only
/// guards topology (manifold, no duplicate edge); a non-convex flip is
/// topologically legal but geometrically a fold, which this rejects.
template <typename PointsPolicy, typename Index>
inline auto flip_keeps_orientation(const tf::points<PointsPolicy> &points,
                                   Index va0, Index va1, Index va2, Index vb2)
    -> bool {
  using Real = tf::coordinate_type<PointsPolicy>;
  auto P = [&](Index v) {
    auto p = points[v];
    return tf::point<Real, 3>{Real(p[0]), Real(p[1]), Real(p[2])};
  };
  auto nrm = [](const tf::point<Real, 3> &p, const tf::point<Real, 3> &q,
                const tf::point<Real, 3> &r) {
    return tf::cross(tf::make_vector(q[0] - p[0], q[1] - p[1], q[2] - p[2]),
                     tf::make_vector(r[0] - p[0], r[1] - p[1], r[2] - p[2]));
  };
  auto pa0 = P(va0), pa1 = P(va1), pa2 = P(va2), pb2 = P(vb2);
  auto n_old_a = nrm(pa0, pa1, pa2); // old (va0,va1,va2)
  auto n_old_b = nrm(pa1, pa0, pb2); // old (va1,va0,vb2)
  auto n_new_a = nrm(pa2, pb2, pa1); // new (va2,vb2,va1)
  auto n_new_b = nrm(pb2, pa2, pa0); // new (vb2,va2,va0)
  return tf::dot(n_new_a, n_old_a) > Real(0) &&
         tf::dot(n_new_b, n_old_b) > Real(0);
}

/// Valence-equalizing edge flips. `points` is either a tf::points (geometric
/// fold guard active) or tf::none (no guard, valence only). `mask` is either a
/// feature_mask (skip feature edges/vertices) or tf::none.
template <typename Index, typename PointsOrNone, typename MaskOrNone>
auto optimize_valence_impl(tf::half_edges<Index> &he, const PointsOrNone &points,
                           tf::buffer<std::int8_t> &deviation,
                           const MaskOrNone &mask, int iterations) -> Index {
  constexpr bool HasPoints = !std::is_same_v<PointsOrNone, tf::none_t>;
  constexpr bool HasMask = !std::is_same_v<MaskOrNone, tf::none_t>;
  Index n_flipped = 0;
  for (int iter = 0; iter < iterations; ++iter) {
    Index flipped_this_iter = 0;
    for (auto eh : he.edge_handles()) {
      if (!he.is_flip_ok(eh))
        continue;
      if constexpr (HasMask)
        if (mask.is_feature(eh.id()))
          continue;
      auto a0 = he.half_edge_handle(tf::unsafe, eh, false);
      auto a1 = he.next(tf::unsafe, a0);
      auto a2 = he.next(tf::unsafe, a1);
      auto b0 = he.half_edge_handle(tf::unsafe, eh, true);
      auto b1 = he.next(tf::unsafe, b0);
      auto b2 = he.next(tf::unsafe, b1);

      auto va0 = he.start_vertex_handle(tf::unsafe, a0).id();
      auto va1 = he.start_vertex_handle(tf::unsafe, a1).id();
      auto va2 = he.start_vertex_handle(tf::unsafe, a2).id();
      auto vb2 = he.start_vertex_handle(tf::unsafe, b2).id();

      if constexpr (HasMask) {
        // Skip if any involved vertex is on a feature edge.
        if (mask.vertex_type(va0) != vertex_feature_type::regular ||
            mask.vertex_type(va1) != vertex_feature_type::regular ||
            mask.vertex_type(va2) != vertex_feature_type::regular ||
            mask.vertex_type(vb2) != vertex_feature_type::regular)
          continue;
      }

      auto before = std::abs(deviation[va0]) + std::abs(deviation[va1]) +
                    std::abs(deviation[va2]) + std::abs(deviation[vb2]);
      auto after =
          std::abs(deviation[va0] - 1) + std::abs(deviation[va1] - 1) +
          std::abs(deviation[va2] + 1) + std::abs(deviation[vb2] + 1);

      if (after >= before)
        continue;

      if constexpr (HasPoints)
        if (!flip_keeps_orientation(points, va0, va1, va2, vb2))
          continue; // flip would fold the surface

      deviation[va0]--;
      deviation[va1]--;
      deviation[va2]++;
      deviation[vb2]++;
      he.flip(eh);
      ++flipped_this_iter;
    }
    n_flipped += flipped_this_iter;
    if (flipped_this_iter == 0)
      break;
  }
  return n_flipped;
}

} // namespace detail

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges.
///
/// Flips edges to bring vertex valences closer to the ideal (6 for
/// interior vertices, 4 for boundary vertices). Each flip is performed
/// only when it strictly reduces the total valence deviation of the
/// four involved vertices. The deviation buffer is modified in place
/// and remains valid for subsequent calls.
///
/// @param he The half-edge structure (modified in place).
/// @param deviation Per-vertex valence deviation (modified in place).
/// @param iterations The number of passes over all edges.
/// @return The number of edges flipped.
template <typename Index>
auto optimize_valence(tf::half_edges<Index> &he,
                      tf::buffer<std::int8_t> &deviation,
                      int iterations = 1) -> Index {
  return detail::optimize_valence_impl(he, tf::none, deviation, tf::none,
                                       iterations);
}

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges (feature-aware).
///
/// Same as the base overload, but skips flipping feature edges and any flip
/// touching a non-regular (crease/corner) vertex.
template <typename Index>
auto optimize_valence(tf::half_edges<Index> &he,
                      tf::buffer<std::int8_t> &deviation,
                      const feature_mask &mask,
                      int iterations = 1) -> Index {
  return detail::optimize_valence_impl(he, tf::none, deviation, mask,
                                       iterations);
}

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges, with a fold guard.
///
/// Same as the base overload, but additionally rejects any flip that would
/// invert a triangle relative to the surface (a non-convex quad flip). The
/// valence-only overloads check topology via is_flip_ok but no geometry, so on
/// uneven or near-flat regions a valence-improving flip can fold the surface;
/// passing @p points enables the geometric guard that prevents it.
template <typename Index, typename PointsPolicy>
auto optimize_valence(tf::half_edges<Index> &he,
                      const tf::points<PointsPolicy> &points,
                      tf::buffer<std::int8_t> &deviation,
                      int iterations = 1) -> Index {
  return detail::optimize_valence_impl(he, points, deviation, tf::none,
                                       iterations);
}

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges (feature-aware, fold guard).
template <typename Index, typename PointsPolicy>
auto optimize_valence(tf::half_edges<Index> &he,
                      const tf::points<PointsPolicy> &points,
                      tf::buffer<std::int8_t> &deviation,
                      const feature_mask &mask,
                      int iterations = 1) -> Index {
  return detail::optimize_valence_impl(he, points, deviation, mask, iterations);
}

} // namespace tf::remesh

namespace tf {

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges.
///
/// Convenience overload that computes valence deviations internally.
///
/// @param he The half-edge structure (modified in place).
/// @param iterations The number of passes over all edges.
/// @return The number of edges flipped.
template <typename Index>
auto optimize_valence(tf::half_edges<Index> &he, int iterations = 1)
    -> Index {
  auto deviation = tf::remesh::compute_valence_deviations(he);
  return tf::remesh::optimize_valence(he, deviation, iterations);
}

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges (feature-aware).
///
/// Convenience overload that computes valence deviations internally.
template <typename Index>
auto optimize_valence(tf::half_edges<Index> &he,
                      const tf::remesh::feature_mask &mask,
                      int iterations = 1) -> Index {
  auto deviation = tf::remesh::compute_valence_deviations(he);
  return tf::remesh::optimize_valence(he, deviation, mask, iterations);
}

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges, with a fold guard.
///
/// Convenience overload that computes valence deviations internally. Rejects
/// flips that would fold the surface (see the remesh overload taking points).
template <typename Index, typename PointsPolicy>
auto optimize_valence(tf::half_edges<Index> &he,
                      const tf::points<PointsPolicy> &points,
                      int iterations = 1) -> Index {
  auto deviation = tf::remesh::compute_valence_deviations(he);
  return tf::remesh::optimize_valence(he, points, deviation, iterations);
}

/// @ingroup remesh
/// @brief Optimize vertex valence by flipping edges (feature-aware, fold guard).
template <typename Index, typename PointsPolicy>
auto optimize_valence(tf::half_edges<Index> &he,
                      const tf::points<PointsPolicy> &points,
                      const tf::remesh::feature_mask &mask,
                      int iterations = 1) -> Index {
  auto deviation = tf::remesh::compute_valence_deviations(he);
  return tf::remesh::optimize_valence(he, points, deviation, mask, iterations);
}

} // namespace tf
