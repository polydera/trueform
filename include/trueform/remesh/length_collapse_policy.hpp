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

#include "../core/frame_of.hpp"
#include "../core/point.hpp"
#include "../core/points.hpp"
#include "../core/transformed.hpp"
#include "../topology/half_edges.hpp"
#include "./collapse/check.hpp"
#include "./collapse/half_edge_to_collapse.hpp"
#include "./collapse/quadric.hpp"
#include "./make_feature_mask.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Edge-length collapse policy with quadric-optimal placement.
///
/// Orders edges by length (shortest first) and places the surviving
/// vertex at the quadric-optimal position. Rejects collapses that
/// would create edges longer than a maximum length or produce
/// degenerate triangles.
///
/// When a feature angle is set, feature edges receive penalty quadrics
/// and feature-constrained vertices are protected from removal.
///
/// @tparam Real The scalar type.
template <typename Real> struct length_collapse_policy {
  tf::buffer<tf::remesh::quadric> _quadrics;
  tf::remesh::feature_mask _feature_mask;
  Real _threshold;
  Real _max2;
  Real _max_aspect_ratio = 40;
  bool _preserve_boundary = false;
  bool _use_quadric = true;
  double _stabilizer = 0;
  tf::rad<Real> _feature_angle{Real(-1)};
  Real _feature_weight = Real(100);

  explicit length_collapse_policy(Real min_len, Real max_len,
                                  bool preserve_boundary = false,
                                  bool use_quadric = true,
                                  Real max_aspect_ratio = 40,
                                  double stabilizer = 1e-6,
                                  tf::rad<Real> feature_angle = tf::rad<Real>(Real(-1)),
                                  Real feature_weight = Real(100))
      : _threshold(min_len * min_len), _max2(max_len * max_len),
        _max_aspect_ratio(max_aspect_ratio),
        _preserve_boundary(preserve_boundary), _use_quadric(use_quadric),
        _stabilizer(stabilizer),
        _feature_angle(feature_angle), _feature_weight(feature_weight) {}

  auto preserve_boundary() const -> bool { return _preserve_boundary; }
  auto preserve_features() const -> bool { return !_feature_mask.empty(); }
  auto feature_mask() -> tf::remesh::feature_mask & { return _feature_mask; }
  auto feature_mask() const -> const tf::remesh::feature_mask & { return _feature_mask; }

  auto error_threshold() const -> Real { return _threshold; }

  template <typename Index, typename Policy>
  auto init(const tf::half_edges<Index> &he, const tf::points<Policy> &points)
      -> void {
    if (_feature_angle.value >= 0) {
      _feature_mask = tf::make_feature_mask(he, points, _feature_angle);
      if (_use_quadric)
        _quadrics = tf::remesh::compute_vertex_quadrics(
            he, points, _feature_mask, double(_feature_weight));
    } else {
      _feature_mask = {};
      if (_use_quadric)
        _quadrics = tf::remesh::compute_vertex_quadrics(he, points);
    }
  }

  template <typename Index>
  auto half_edge_to_collapse(const tf::half_edges<Index> &he,
                             tf::edge_handle<Index> eh)
      -> tf::half_edge_handle<Index> {
    if (_feature_mask.empty())
      return tf::remesh::half_edge_to_collapse(he, eh);
    return tf::remesh::half_edge_to_collapse(he, eh, _feature_mask);
  }

  template <typename Index, typename Policy>
  auto collapse_error(const tf::half_edges<Index> &he,
                      const tf::points<Policy> &points,
                      tf::half_edge_handle<Index> heh) -> Real {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    return tf::transformed(points[v1] - points[v0], tf::frame_of(points))
        .length2();
  }

  template <typename Index, typename Policy>
  auto collapsed_point(const tf::half_edges<Index> &he,
                       const tf::points<Policy> &points,
                       tf::half_edge_handle<Index> heh) -> tf::point<Real, 3> {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    if (!_use_quadric ||
        he.is_boundary_vertex(v0) || he.is_boundary_vertex(v1))
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
    if (!_feature_mask.empty()) {
      auto v_removed = he.end_vertex_handle(tf::unsafe, heh).id();
      auto eid = he.edge_handle(tf::unsafe, heh).id();
      if (_feature_mask.is_collapse_forbidden(eid, v_removed))
        return false;
    }
    if (_max_aspect_ratio < 0 ||
        _max_aspect_ratio == std::numeric_limits<Real>::max())
      return tf::remesh::is_collapse_allowed_dihedral(he, points, heh, pt,
                                                      tf::none, _max2);
    return tf::remesh::is_collapse_allowed_dihedral(he, points, heh, pt,
                                                    _max_aspect_ratio, _max2);
  }

  template <typename Index, typename Policy>
  auto commit_collapse(Index v0, Index v1, const tf::point<Real, 3> &pt,
                       tf::points<Policy> &points) -> void {
    if (_use_quadric)
      tf::remesh::commit_collapse_quadric<Real>(_quadrics, v0, v1, pt, points);
    else
      points[v0] = pt;
  }
};

} // namespace tf
