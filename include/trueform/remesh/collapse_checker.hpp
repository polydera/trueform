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
/// Wraps is_collapse_allowed_dihedral with pluggable max_aspect_ratio
/// and max_edge_length providers. Each can be:
///   - tf::none_t — check disabled (zero-cost via if constexpr)
///   - a callable (half_edges, points, edge_handle) → Real
///
/// Use make_collapse_checker to construct from scalars, callables, or none.
template <typename AspectRatioFn, typename EdgeLengthFn>
struct collapse_checker {
  AspectRatioFn _max_aspect_ratio;
  EdgeLengthFn _max_edge_length;

  collapse_checker(AspectRatioFn ar, EdgeLengthFn el)
      : _max_aspect_ratio(std::move(ar)), _max_edge_length(std::move(el)) {}

  template <typename Index, typename PointsPolicy, typename Real>
  auto operator()(const tf::half_edges<Index> &he,
                  const tf::points<PointsPolicy> &points,
                  tf::half_edge_handle<Index> heh,
                  const tf::point<Real, 3> &pt) const -> bool {
    constexpr bool has_ar = !std::is_same_v<AspectRatioFn, tf::none_t>;
    constexpr bool has_el = !std::is_same_v<EdgeLengthFn, tf::none_t>;

    if constexpr (!has_ar && !has_el) {
      return tf::remesh::is_collapse_allowed_dihedral(he, points, heh, pt);
    } else if constexpr (has_ar && !has_el) {
      auto eh = he.edge_handle(tf::unsafe, heh);
      return tf::remesh::is_collapse_allowed_dihedral(
          he, points, heh, pt, _max_aspect_ratio(he, points, eh), tf::none);
    } else if constexpr (!has_ar && has_el) {
      auto eh = he.edge_handle(tf::unsafe, heh);
      return tf::remesh::is_collapse_allowed_dihedral(
          he, points, heh, pt, tf::none, _max_edge_length(he, points, eh));
    } else {
      auto eh = he.edge_handle(tf::unsafe, heh);
      return tf::remesh::is_collapse_allowed_dihedral(
          he, points, heh, pt, _max_aspect_ratio(he, points, eh),
          _max_edge_length(he, points, eh));
    }
  }
};

/// @brief Create a collapse check from scalars, callables, or none.
///
/// Each argument can be:
///   - tf::none — disable the check
///   - a scalar (int, float, double, ...) — constant threshold for all edges
///   - a callable (half_edges, points, edge_handle) → Real — per-edge threshold
template <typename Real, typename AR = const tf::none_t &, typename EL = const tf::none_t&>
auto make_collapse_checker(AR &&max_aspect_ratio = tf::none,
                           EL &&max_edge_length = tf::none) {
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
  auto ar = wrap(std::forward<AR>(max_aspect_ratio));
  auto el = wrap(std::forward<EL>(max_edge_length));
  return collapse_checker<decltype(ar), decltype(el)>(std::move(ar),
                                                     std::move(el));
}

} // namespace tf
