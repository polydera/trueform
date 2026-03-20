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

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/frame_of.hpp"
#include "../core/points_buffer.hpp"
#include "../core/transformed.hpp"
#include "./split_edges.hpp"
#include "./split_handler.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Split edges longer than a maximum length (frame-aware).
///
/// Constructs a split_handler that scores edges by length² / max_length²
/// and delegates to split_edges.
template <typename Index, typename Real, std::size_t Dims, typename FramePolicy>
auto split_long_edges(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points,
                      const tf::frame_like<Dims, FramePolicy> &frame,
                      Real max_length, bool preserve_boundary = true)
    -> Index {
  Real max2 = max_length * max_length;
  auto handler = tf::make_split_handler<Real>(
      [max2, &frame](const auto &he, const auto &points, auto heh) -> Real {
        auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
        auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
        auto len2 =
            tf::transformed(points[v1] - points[v0], frame).length2();
        return len2 / max2;
      },
      preserve_boundary);
  return tf::split_edges(he, points, frame, handler);
}

/// @ingroup remesh
/// @brief Split edges longer than a maximum length.
/// @overload
template <typename Index, typename Real, std::size_t Dims>
auto split_long_edges(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points, Real max_length,
                      bool preserve_boundary = true) -> Index {
  return tf::split_long_edges(he, points, tf::identity_frame<Real, Dims>{},
                              max_length, preserve_boundary);
}

/// @ingroup remesh
/// @brief Split edges longer than a maximum length.
///
/// Creates a new points buffer with the original and created vertices.
/// Extracts the coordinate frame from points if tagged.
template <typename Index, typename PointsPolicy>
auto split_long_edges(tf::half_edges<Index> &he,
                      const tf::points<PointsPolicy> &points,
                      tf::coordinate_type<PointsPolicy> max_length,
                      bool preserve_boundary = true)
    -> tf::points_buffer<tf::coordinate_type<PointsPolicy>,
                         tf::coordinate_dims_v<PointsPolicy>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<PointsPolicy>;
  auto frame = tf::frame_of(points);
  tf::points_buffer<Real, Dims> buf;
  buf.allocate(points.size());
  tf::parallel_copy(points, buf.points());
  tf::split_long_edges(he, buf, frame, max_length, preserve_boundary);
  return buf;
}

} // namespace tf
