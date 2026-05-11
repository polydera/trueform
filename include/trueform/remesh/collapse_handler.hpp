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

#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/points.hpp"
#include "../topology/half_edges.hpp"
#include "./collapse/half_edge_to_collapse.hpp"
#include "./collapse/quadric.hpp"
#include "./collapse_config.hpp"

#include <type_traits>
#include <utility>

namespace tf {

template <typename Real, typename ScoreFn, typename AllowedFn,
          typename FeatureHandler = tf::none_t>
struct collapse_handler {
  ScoreFn _score;
  AllowedFn _allowed;
  tf::buffer<tf::remesh::quadric> _quadrics;
  tf::collapse_config<Real> _config;
  [[no_unique_address]] FeatureHandler _features;

  collapse_handler(ScoreFn score, AllowedFn allowed,
                   const tf::collapse_config<Real> &config = {},
                   FeatureHandler features = {})
      : _score(std::move(score)), _allowed(std::move(allowed)),
        _config(config), _features(std::move(features)) {}

  auto preserve_boundary() const -> bool { return _config.preserve_boundary; }
  auto parallel() const -> bool { return _config.parallel; }
  auto preserve_features() const -> bool {
    if constexpr (std::is_same_v<FeatureHandler, tf::none_t>)
      return false;
    else
      return !_features.empty();
  }

  auto error_threshold() const -> Real { return Real(1); }

  template <typename Index, typename Policy>
  auto init(const tf::half_edges<Index> &he, const tf::points<Policy> &points)
      -> void {
    if (!_config.use_quadric)
      return;
    if constexpr (std::is_same_v<FeatureHandler, tf::none_t>) {
      _quadrics = tf::remesh::compute_vertex_quadrics(he, points);
    } else {
      if (_features.empty())
        _quadrics = tf::remesh::compute_vertex_quadrics(he, points);
      else
        _quadrics = tf::remesh::compute_vertex_quadrics(
            he, points, _features, double(_config.feature_weight));
    }
  }

  template <typename Index>
  auto half_edge_to_collapse(const tf::half_edges<Index> &he,
                             tf::edge_handle<Index> eh)
      -> tf::half_edge_handle<Index> {
    if constexpr (std::is_same_v<FeatureHandler, tf::none_t>) {
      return tf::remesh::half_edge_to_collapse(he, eh);
    } else {
      if (_features.empty())
        return tf::remesh::half_edge_to_collapse(he, eh);
      return tf::remesh::half_edge_to_collapse(he, eh, _features);
    }
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
    if (!_config.use_quadric || he.is_boundary_vertex(v0) ||
        he.is_boundary_vertex(v1))
      return points[v0];
    if constexpr (!std::is_same_v<FeatureHandler, tf::none_t>) {
      if (!_features.empty() &&
          (_features.is_corner(v0) || _features.is_crease(v0)))
        return points[v0];
    }
    return tf::remesh::collapsed_point_quadric<Real>(_quadrics, points, v0, v1,
                                                     _config.stabilizer);
  }

  template <typename Index, typename Policy>
  auto is_collapse_allowed(const tf::half_edges<Index> &he,
                           const tf::points<Policy> &points,
                           tf::half_edge_handle<Index> heh,
                           const tf::point<Real, 3> &pt) -> bool {
    if constexpr (!std::is_same_v<FeatureHandler, tf::none_t>) {
      if (!_features.empty()) {
        auto v_removed = he.end_vertex_handle(tf::unsafe, heh).id();
        auto eid = he.edge_handle(tf::unsafe, heh).id();
        if (_features.is_collapse_forbidden(eid, v_removed))
          return false;
      }
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

template <typename Real, typename ScoreFn, typename AllowedFn>
auto make_collapse_handler(ScoreFn &&score, AllowedFn &&allowed,
                           const tf::collapse_config<Real> &config = {}) {
  return collapse_handler<Real, std::decay_t<ScoreFn>, std::decay_t<AllowedFn>>(
      std::forward<ScoreFn>(score), std::forward<AllowedFn>(allowed), config);
}

template <typename Real, typename FeatureHandler, typename ScoreFn,
          typename AllowedFn>
auto make_collapse_handler(ScoreFn &&score, AllowedFn &&allowed,
                           FeatureHandler &&features,
                           const tf::collapse_config<Real> &config) {
  return collapse_handler<Real, std::decay_t<ScoreFn>, std::decay_t<AllowedFn>,
                          std::decay_t<FeatureHandler>>(
      std::forward<ScoreFn>(score), std::forward<AllowedFn>(allowed), config,
      std::forward<FeatureHandler>(features));
}

} // namespace tf
