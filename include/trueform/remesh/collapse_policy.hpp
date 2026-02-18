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

#include "./collapse/check.hpp"
#include "./collapse/quadric.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Quadric error metric collapse policy (Garland & Heckbert).
///
/// Maintains per-vertex quadric error matrices. Edge collapse error is
/// the square root of the quadric error at the optimal collapse point.
/// Rejects collapses that would cause normal flips or degenerate triangles.
///
/// @tparam Real The scalar type.
template <typename Real> struct collapse_policy {
  tf::buffer<tf::remesh::quadric> _quadrics;
  Real _max_aspect_ratio;
  bool _preserve_boundary = false;
  double _stabilizer = 0;

  explicit collapse_policy(Real max_aspect_ratio = Real(40),
                           bool preserve_boundary = false,
                           double stabilizer = 1e-3)
      : _max_aspect_ratio(max_aspect_ratio),
        _preserve_boundary(preserve_boundary), _stabilizer(stabilizer) {}

  auto preserve_boundary() const -> bool { return _preserve_boundary; }

  template <typename Index, typename Policy>
  auto init(const tf::half_edges<Index> &he, const tf::points<Policy> &points)
      -> void {
    _quadrics = tf::remesh::compute_vertex_quadrics(he, points);
  }

  template <typename Index>
  auto half_edge_to_collapse(const tf::half_edges<Index> &he,
                             tf::edge_handle<Index> eh)
      -> tf::half_edge_handle<Index> {
    auto heh0 = he.half_edge_handle(tf::unsafe, eh, false);
    auto v0 = he.start_vertex_handle(tf::unsafe, heh0).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh0).id();
    if (he.is_boundary_vertex(v1) && !he.is_boundary_vertex(v0))
      return he.opposite(tf::unsafe, heh0);
    return heh0;
  }

  template <typename Index, typename Policy>
  auto collapse_error(const tf::half_edges<Index> &he,
                      const tf::points<Policy> &points,
                      tf::half_edge_handle<Index> heh) -> Real {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    tf::remesh::quadric q = _quadrics[v0];
    q += _quadrics[v1];
    if (he.is_boundary_vertex(v0) || he.is_boundary_vertex(v1))
      return tf::sqrt(q.evaluate(points[v0]));
    if (auto opt = tf::remesh::solve_optimal_quadric<Real>(q, _stabilizer))
      return tf::sqrt(q.evaluate(*opt));
    auto e0 = q.evaluate(points[v0]);
    auto e1 = q.evaluate(points[v1]);
    auto mid = tf::make_point((points[v0][0] + points[v1][0]) / 2,
                              (points[v0][1] + points[v1][1]) / 2,
                              (points[v0][2] + points[v1][2]) / 2);
    return tf::sqrt(std::min({e0, e1, q.evaluate(mid)}));
  }

  template <typename Index, typename Policy>
  auto collapsed_point(const tf::half_edges<Index> &he,
                       const tf::points<Policy> &points,
                       tf::half_edge_handle<Index> heh) -> tf::point<Real, 3> {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    if (he.is_boundary_vertex(v0) || he.is_boundary_vertex(v1))
      return tf::make_point(Real(points[v0][0]), Real(points[v0][1]),
                            Real(points[v0][2]));
    return tf::remesh::collapsed_point_quadric<Real>(_quadrics, points, v0, v1,
                                                     _stabilizer);
  }

  template <typename Index, typename Policy>
  auto is_collapse_allowed(const tf::half_edges<Index> &he,
                           const tf::points<Policy> &points,
                           tf::half_edge_handle<Index> heh,
                           const tf::point<Real, 3> &pt) -> bool {
    if (_max_aspect_ratio < 0 ||
        _max_aspect_ratio == std::numeric_limits<Real>::max())
      return tf::remesh::is_collapse_allowed(he, points, heh, pt, tf::none,
                                             tf::none);
    return tf::remesh::is_collapse_allowed(he, points, heh, pt,
                                           _max_aspect_ratio, tf::none);
  }

  template <typename Index, typename Policy>
  auto commit_collapse(Index v0, Index v1, const tf::point<Real, 3> &pt,
                       tf::points<Policy> &points) -> void {
    tf::remesh::commit_collapse_quadric<Real>(_quadrics, v0, v1, pt, points);
  }
};

} // namespace tf
