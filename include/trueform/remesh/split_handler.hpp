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

#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../topology/half_edges.hpp"

#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup remesh
/// @brief Generic split handler with pluggable scoring and an optional
/// propagation tracker.
///
/// The scoring function maps (he, points, heh) → Real. Values > 1
/// trigger a split.
///
/// The optional Tracker is notified after each split pass with the
/// parent-of-new-element arrays so it can propagate per-element data
/// (e.g. feature_mask edge flags, per-face labels).
///
/// @tparam Real    The scalar type.
/// @tparam ScoreFn Scoring callable: (half_edges, points, half_edge_handle) → Real.
/// @tparam Tracker Optional propagation observer; defaults to tf::none_t.
template <typename Real, typename ScoreFn, typename Tracker = tf::none_t>
struct split_handler {
  ScoreFn _score;
  bool _preserve_boundary = true;
  int _max_iterations = 3;
  Tracker _tracker;

  split_handler(ScoreFn score, bool preserve_boundary = true,
                int max_iterations = 3, Tracker tracker = {})
      : _score(std::move(score)), _preserve_boundary(preserve_boundary),
        _max_iterations(max_iterations), _tracker(std::move(tracker)) {}

  auto preserve_boundary() const -> bool { return _preserve_boundary; }
  auto max_iterations() const -> int { return _max_iterations; }

  template <typename Index>
  auto max_length2(tf::edge_handle<Index>) const -> Real { return Real(1); }

  template <typename Index>
  auto should_skip(tf::edge_handle<Index>) const -> bool { return false; }

  template <typename Index, typename PointsPolicy>
  auto distance2(const tf::half_edges<Index> &he,
                 const tf::points<PointsPolicy> &points,
                 tf::half_edge_handle<Index> heh) const -> Real {
    return _score(he, points, heh);
  }

  template <typename Index, typename PointsPolicy>
  auto interpolate(const tf::half_edges<Index> &he,
                   const tf::points<PointsPolicy> &points,
                   tf::half_edge_handle<Index> heh) const {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    using coord_t = tf::coordinate_type<PointsPolicy>;
    return points[v0] +
           (points[v1].as_vector_view() - points[v0].as_vector_view()) *
               coord_t(0.5);
  }

  template <typename Index, typename PointsPolicy, typename FaceBuf,
            typename EdgeBuf>
  auto on_split_done(const tf::half_edges<Index> &he,
                     const tf::points<PointsPolicy> &points,
                     const FaceBuf &parent_face_for_new,
                     const EdgeBuf &parent_edge_for_new) const -> void {
    if constexpr (!std::is_same_v<Tracker, tf::none_t>)
      _tracker.update(he, points, parent_face_for_new, parent_edge_for_new);
  }
};

/// @brief Create a split handler from a scoring lambda (no tracker).
template <typename Real, typename ScoreFn>
auto make_split_handler(ScoreFn &&score, bool preserve_boundary = true,
                        int max_iterations = 3) {
  return split_handler<Real, std::decay_t<ScoreFn>>(
      std::forward<ScoreFn>(score), preserve_boundary, max_iterations);
}

/// @brief Create a split handler with an attached tracker.
template <typename Real, typename Tracker, typename ScoreFn>
auto make_split_handler(ScoreFn &&score, Tracker &&tracker,
                        bool preserve_boundary = true,
                        int max_iterations = 3) {
  return split_handler<Real, std::decay_t<ScoreFn>, std::decay_t<Tracker>>(
      std::forward<ScoreFn>(score), preserve_boundary, max_iterations,
      std::forward<Tracker>(tracker));
}

} // namespace tf
