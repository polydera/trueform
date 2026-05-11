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

#include "../core/frame_like.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/points_buffer.hpp"
#include "../core/views/drop.hpp"
#include "./split/half_edge_splitter.hpp"
#include "./split_handler.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Split edges using a split_handler (frame-aware).
///
/// Iterates until the handler reports no more edges to split or
/// max_iterations is reached.
template <typename Index, typename Real, std::size_t Dims,
          typename FramePolicy, typename ScoreFn, typename Tracker>
auto split_edges(tf::half_edges<Index> &he,
                 tf::points_buffer<Real, Dims> &points,
                 const tf::frame_like<Dims, FramePolicy> &frame,
                 const tf::split_handler<Real, ScoreFn, Tracker> &handler)
    -> Index {
  tf::remesh::half_edge_splitter<Index, Real, Dims> splitter;
  Index total_split = 0;
  int iter = 0;
  while (iter++ < handler.max_iterations()) {
    auto tagged = points.points() | tf::tag(frame);
    Index n_faces =
        splitter.split(tagged, he, handler, handler.preserve_boundary());
    const auto &created = splitter.created_point_data();
    if (created.empty())
      break;
    total_split += Index(created.size());
    auto old_n = points.size();
    points.reallocate(old_n + created.size());
    tf::parallel_copy(
        tf::make_range(created.data_buffer()),
        tf::drop(tf::make_range(points.data_buffer()), old_n * Dims));
    he.rebuild_handles(n_faces, Index(points.size()));
    handler.on_split_done(he, splitter.parent_face_for_new(),
                          splitter.parent_edge_for_new());
  }
  return total_split;
}

/// @brief Split edges using a split_handler.
/// @overload
template <typename Index, typename Real, std::size_t Dims, typename ScoreFn,
          typename Tracker>
auto split_edges(tf::half_edges<Index> &he,
                 tf::points_buffer<Real, Dims> &points,
                 const tf::split_handler<Real, ScoreFn, Tracker> &handler)
    -> Index {
  return tf::split_edges(he, points, tf::identity_frame<Real, Dims>{},
                         handler);
}

} // namespace tf
