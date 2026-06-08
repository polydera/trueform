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
#include "./collapse/check.hpp"

#include <type_traits>

namespace tf {

/// @ingroup remesh
/// @brief Geometric acceptance check for edge collapse.
///
/// Wraps is_collapse_allowed_dihedral with pluggable min_quality
/// and max_edge_length providers. Each can be:
///   - tf::none_t — check disabled (zero-cost via if constexpr)
///   - a callable (half_edges, points, edge_handle) → Real
///
/// min_quality is the worst triangle quality (in [0,1], 1 = equilateral)
/// allowed after a collapse: a collapse is permitted iff the worst new quality
/// is >= min(min_quality, worst old quality). 0 = never worsen, >0 = quality
/// floor, <0 disables.
///
/// Use make_collapse_checker to construct from scalars, callables, or none.
///
/// The check_normals runtime flag adds the per-face old-vs-new normal-flip
/// guard (CGAL's Bounded_normal_change) on top of the always-on dihedral fold
/// check.
template <typename MinQualityFn, typename EdgeLengthFn>
struct collapse_checker {
  MinQualityFn _min_quality;
  EdgeLengthFn _max_edge_length;
  bool _check_normals = false;

  collapse_checker(MinQualityFn mq, EdgeLengthFn el, bool check_normals = false)
      : _min_quality(std::move(mq)), _max_edge_length(std::move(el)),
        _check_normals(check_normals) {}

  template <typename Index, typename PointsPolicy, typename Real>
  auto operator()(const tf::half_edges<Index> &he,
                  const tf::points<PointsPolicy> &points,
                  tf::half_edge_handle<Index> heh,
                  const tf::point<Real, 3> &pt) const -> bool {
    constexpr bool has_q = !std::is_same_v<MinQualityFn, tf::none_t>;
    constexpr bool has_el = !std::is_same_v<EdgeLengthFn, tf::none_t>;

    if constexpr (!has_q && !has_el) {
      return tf::remesh::is_collapse_allowed_dihedral(he, points, heh, pt,
                                                      tf::none, tf::none,
                                                      _check_normals);
    } else if constexpr (has_q && !has_el) {
      auto eh = he.edge_handle(tf::unsafe, heh);
      return tf::remesh::is_collapse_allowed_dihedral(
          he, points, heh, pt, _min_quality(he, points, eh), tf::none,
          _check_normals);
    } else if constexpr (!has_q && has_el) {
      auto eh = he.edge_handle(tf::unsafe, heh);
      return tf::remesh::is_collapse_allowed_dihedral(
          he, points, heh, pt, tf::none, _max_edge_length(he, points, eh),
          _check_normals);
    } else {
      auto eh = he.edge_handle(tf::unsafe, heh);
      return tf::remesh::is_collapse_allowed_dihedral(
          he, points, heh, pt, _min_quality(he, points, eh),
          _max_edge_length(he, points, eh), _check_normals);
    }
  }
};

/// @brief Create a collapse check from scalars, callables, or none.
///
/// Each argument can be:
///   - tf::none — disable the check
///   - a scalar (int, float, double, ...) — constant threshold for all edges
///   - a callable (half_edges, points, edge_handle) → Real — per-edge threshold
/// @param check_normals When true, the checker also rejects collapses that
///                    invert any surviving face's normal (CGAL
///                    Bounded_normal_change). Runtime flag, default false.
template <typename Real, typename MQ = const tf::none_t &,
          typename EL = const tf::none_t &>
auto make_collapse_checker(MQ &&min_quality = tf::none,
                           EL &&max_edge_length = tf::none,
                           bool check_normals = false) {
  auto wrap = [](auto &&v) {
    using D = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<D, tf::none_t>) {
      return tf::none;
    } else if constexpr (std::is_arithmetic_v<D>) {
      Real val = Real(v);
      return [val](const auto &, const auto &, const auto &) -> Real {
        return val;
      };
    } else {
      return std::forward<decltype(v)>(v);
    }
  };
  auto mq = wrap(std::forward<MQ>(min_quality));
  auto el = wrap(std::forward<EL>(max_edge_length));
  return collapse_checker<decltype(mq), decltype(el)>(std::move(mq),
                                                      std::move(el),
                                                      check_normals);
}

} // namespace tf
