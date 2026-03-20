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

#include "../core/point.hpp"
#include "../core/points.hpp"
#include "../topology/half_edges.hpp"
#include "./collapse/half_edge_to_collapse.hpp"
#include "./collapse/quadric.hpp"
#include "./collapse_config.hpp"
#include "./make_feature_mask.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Generic collapse handler with pluggable scoring and acceptance.
///
/// Holds shared collapse machinery (quadrics, feature mask, point
/// placement). Two callables control the policy:
///
///   ScoreFn:   (half_edges, points, half_edge_handle) → Real
///              Returns a value in [0, ∞). Values < 1 are eligible
///              for collapse; the value is the priority (0 = highest).
///
///   AllowedFn: (half_edges, points, half_edge_handle, point) → bool
///              Returns true if the collapse is geometrically acceptable.
///
/// The error_threshold is always 1.
///
/// @tparam Real      The scalar type.
/// @tparam ScoreFn   Scoring callable.
/// @tparam AllowedFn Acceptance callable.
template <typename Real, typename ScoreFn, typename AllowedFn>
struct collapse_handler {
  ScoreFn _score;
  AllowedFn _allowed;
  tf::buffer<tf::remesh::quadric> _quadrics;
  tf::remesh::feature_mask _feature_mask;
  tf::collapse_config<Real> _config;

  collapse_handler(ScoreFn score, AllowedFn allowed,
                   const tf::collapse_config<Real> &config = {})
      : _score(std::move(score)), _allowed(std::move(allowed)),
        _config(config) {}

  auto preserve_boundary() const -> bool { return _config.preserve_boundary; }
  auto parallel() const -> bool { return _config.parallel; }
  auto preserve_features() const -> bool { return !_feature_mask.empty(); }
  auto feature_mask() -> tf::remesh::feature_mask & { return _feature_mask; }
  auto feature_mask() const -> const tf::remesh::feature_mask & {
    return _feature_mask;
  }

  auto error_threshold() const -> Real { return Real(1); }

  template <typename Index, typename Policy>
  auto init(const tf::half_edges<Index> &he, const tf::points<Policy> &points)
      -> void {
    if (_config.feature_angle.value >= 0) {
      _feature_mask = tf::make_feature_mask(he, points, _config.feature_angle);
      if (_config.use_quadric)
        _quadrics = tf::remesh::compute_vertex_quadrics(
            he, points, _feature_mask, double(_config.feature_weight));
    } else {
      _feature_mask = {};
      if (_config.use_quadric)
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
    return _score(he, points, heh, *this);
  }

  template <typename Index, typename Policy>
  auto collapsed_point(const tf::half_edges<Index> &he,
                       const tf::points<Policy> &points,
                       tf::half_edge_handle<Index> heh) -> tf::point<Real, 3> {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    if (!_config.use_quadric ||
        he.is_boundary_vertex(v0) || he.is_boundary_vertex(v1))
      return tf::make_point(Real(points[v0][0]), Real(points[v0][1]),
                            Real(points[v0][2]));
    return tf::remesh::collapsed_point_quadric<Real>(
        _quadrics, points, v0, v1, _config.stabilizer);
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
    return _allowed(he, points, heh, pt);
  }

  template <typename Index, typename Policy>
  auto commit_collapse(Index v0, Index v1, const tf::point<Real, 3> &pt,
                       tf::points<Policy> &points) -> void {
    if (_config.use_quadric)
      tf::remesh::commit_collapse_quadric<Real>(_quadrics, v0, v1, pt, points);
    else
      points[v0] = pt;
  }
};

/// @brief Create a collapse handler from scoring and acceptance lambdas.
template <typename Real, typename ScoreFn, typename AllowedFn>
auto make_collapse_handler(ScoreFn &&score, AllowedFn &&allowed,
                           const tf::collapse_config<Real> &config = {}) {
  return collapse_handler<Real, std::decay_t<ScoreFn>,
                          std::decay_t<AllowedFn>>(
      std::forward<ScoreFn>(score), std::forward<AllowedFn>(allowed), config);
}

} // namespace tf
