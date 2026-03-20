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
#include "../core/frame_of.hpp"
#include "../core/transformed.hpp"
#include "./collapse_checker.hpp"
#include "./collapse_edges.hpp"
#include "./collapse_handler.hpp"
#include "./length_collapse_config.hpp"

namespace tf {

// ---------------------------------------------------------------------------
// Exhaustion mode — collapse all edges shorter than min_len
// ---------------------------------------------------------------------------

/// @ingroup remesh
/// @brief Collapse edges shorter than min_len.
template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config = {}) -> Index {
  using Real = tf::coordinate_type<PointsPolicy>;
  Real min2 = min_len * min_len;
  Real max2 = config.max_length * config.max_length;

  auto score = [min2](const auto &he, const auto &points, auto heh,
                      const auto &) -> Real {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    auto len2 =
        tf::transformed(points[v1] - points[v0], tf::frame_of(points))
            .length2();
    return len2 / min2;
  };

  auto checker = tf::make_collapse_checker<Real>(config.max_aspect_ratio, max2);
  auto handler =
      tf::make_collapse_handler<Real>(score, checker, config);
  return tf::collapse_edges(he, points, handler);
}

template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config) -> Index {
  return tf::collapse_short_edges(he, points, min_len, config);
}

// ---------------------------------------------------------------------------
// Target mode — collapse shortest edges until target face count is reached
// ---------------------------------------------------------------------------

/// @ingroup remesh
/// @brief Collapse short edges to reach a target face count.
template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config = {}) -> Index {
  using Real = tf::coordinate_type<PointsPolicy>;
  Real max2 = config.max_length * config.max_length;

  auto score = [](const auto &he, const auto &points, auto heh,
                   const auto &) -> Real {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    return tf::transformed(points[v1] - points[v0], tf::frame_of(points))
        .length2();
  };

  auto checker = tf::make_collapse_checker<Real>(config.max_aspect_ratio, max2);
  auto handler =
      tf::make_collapse_handler<Real>(score, checker, config);
  return tf::collapse_edges(he, points, handler, target_faces);
}

template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config) -> Index {
  return tf::collapse_short_edges(he, points, target_faces, config);
}

} // namespace tf
