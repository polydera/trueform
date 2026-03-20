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
#include "../reindex/points.hpp"
#include "./collapse_checker.hpp"
#include "./collapse_edges.hpp"
#include "./collapse_handler.hpp"
#include "./optimize_valence.hpp"
#include "./remesh_config.hpp"
#include "./split_long_edges.hpp"
#include "./tangential_relaxation.hpp"
#include "./valence_deviation.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Standard isotropic remeshing (frame-aware).
///
/// Each iteration: split edges > 4/3*L, collapse edges < 4/5*L,
/// optimize valence via edge flips, then tangential relaxation.
/// Edge lengths are measured in the given coordinate frame.
template <typename Index, typename Real, std::size_t Dims, typename FramePolicy>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points,
                      const tf::frame_like<Dims, FramePolicy> &frame,
                      const tf::remesh_config<Real> &config) -> void {
  Real high = Real(4) / 3 * config.target_length;
  Real low = Real(4) / 5 * config.target_length;
  Real high2 = high * high;
  Real low2 = low * low;

  auto collapse_score = [low2](const auto &he, const auto &points, auto heh,
                               const auto &) -> Real {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    return tf::transformed(points[v1] - points[v0], tf::frame_of(points))
               .length2() /
           low2;
  };
  auto checker = tf::make_collapse_checker<Real>(config.max_aspect_ratio, high2);
  auto collapse_handler =
      tf::make_collapse_handler<Real>(collapse_score, checker, config);

  tf::points_buffer<double, Dims> old_pos;

  for (int i = 0; i < config.iterations; ++i) {
    tf::split_long_edges(he, points, frame, high, config.preserve_boundary);

    {
      auto tagged = points.points() | tf::tag(frame);
      tf::collapse_edges(he, tagged, collapse_handler);
    }

    auto [face_im, vert_im, edge_im] = he.compact();
    points = tf::reindexed(points.points(), vert_im);

    auto deviation = tf::remesh::compute_valence_deviations(he);

    if (collapse_handler.preserve_features()) {
      collapse_handler.feature_mask().compact(edge_im, vert_im);
      tf::remesh::optimize_valence(he, deviation,
                                   collapse_handler.feature_mask(), 1);
      tf::remesh::tangential_relaxation(he, points.points(), old_pos,
                                        collapse_handler.feature_mask(),
                                        config.relaxation_iters, config.lambda);
    } else {
      tf::remesh::optimize_valence(he, deviation, 1);
      tf::remesh::tangential_relaxation(he, points.points(), old_pos,
                                        config.relaxation_iters, config.lambda);
    }
  }
}

/// @ingroup remesh
/// @brief Standard isotropic remeshing.
/// @overload
template <typename Index, typename Real, std::size_t Dims>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points,
                      const tf::remesh_config<Real> &config) -> void {
  tf::isotropic_remesh(he, points, tf::identity_frame<Real, Dims>{}, config);
}

/// @brief Overload with just target edge length.
template <typename Index, typename Real, std::size_t Dims>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points, Real target_length)
    -> void {
  tf::isotropic_remesh(he, points, tf::make_remesh_config(target_length));
}

/// @brief Overload taking const points, returns a new points buffer.
template <typename Index, typename PointsPolicy>
auto isotropic_remesh(
    tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    const tf::remesh_config<tf::coordinate_type<PointsPolicy>> &config)
    -> tf::points_buffer<tf::coordinate_type<PointsPolicy>,
                         tf::coordinate_dims_v<PointsPolicy>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<PointsPolicy>;
  auto frame = tf::frame_of(points);
  tf::points_buffer<Real, Dims> buf;
  buf.allocate(points.size());
  tf::parallel_copy(points, buf.points());
  tf::isotropic_remesh(he, buf, frame, config);
  return buf;
}

/// @brief Overload taking const points with just target length.
template <typename Index, typename PointsPolicy>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      const tf::points<PointsPolicy> &points,
                      tf::coordinate_type<PointsPolicy> target_length)
    -> tf::points_buffer<tf::coordinate_type<PointsPolicy>,
                         tf::coordinate_dims_v<PointsPolicy>> {
  return tf::isotropic_remesh(he, points,
                              tf::make_remesh_config(target_length));
}

} // namespace tf
