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
#include "../core/range.hpp"
#include "../core/views/drop.hpp"
#include "./split/half_edge_splitter.hpp"
#include "./split/length_split_handler.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Split edges longer than a maximum length (frame-aware).
///
/// Iterates until no edges exceed the threshold. Grows the points buffer
/// in place and rebuilds the half-edge structure between passes.
/// Edge lengths are measured in the given coordinate frame.
template <typename Index, typename Real, std::size_t Dims,
          typename FramePolicy>
auto split_long_edges(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points,
                      const tf::frame_like<Dims, FramePolicy> &frame,
                      Real max_length,
                      bool preserve_boundary = false) -> Index {
  tf::remesh::half_edge_splitter<Index, Real, Dims> splitter;
  tf::remesh::length_split_handler<Real, Dims> handler(max_length);
  Index total_split = 0;
  while (true) {
    auto tagged = points.points() | tf::tag(frame);
    Index n_faces =
        splitter.split(tagged, he, handler, preserve_boundary);
    const auto &created = splitter.created_point_data();
    if (created.empty())
      break;
    total_split += Index(created.size());
    auto old_n = points.size();
    points.reallocate(old_n + created.size());
    tf::parallel_copy(tf::make_range(created.data_buffer()),
                      tf::drop(tf::make_range(points.data_buffer()),
                               old_n * Dims));
    he.rebuild_handles(n_faces, Index(points.size()));
  }
  return total_split;
}

/// @ingroup remesh
/// @brief Split edges longer than a maximum length.
/// @overload
template <typename Index, typename Real, std::size_t Dims>
auto split_long_edges(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points, Real max_length,
                      bool preserve_boundary = false) -> Index {
  return tf::split_long_edges(he, points, tf::identity_frame<Real, Dims>{},
                              max_length, preserve_boundary);
}

/// @ingroup remesh
/// @brief Split edges longer than a maximum length.
///
/// Creates a new points buffer with the original and created vertices.
/// Extracts the coordinate frame from points if tagged.
template <typename Index, typename PointsPolicy>
auto split_long_edges(
    tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    tf::coordinate_type<PointsPolicy> max_length,
    bool preserve_boundary = false)
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
