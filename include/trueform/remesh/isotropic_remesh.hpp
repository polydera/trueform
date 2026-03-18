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
#include "./collapse/collapse_to_exhaustion.hpp"
#include "./collapse/collapse_to_exhaustion_parallel.hpp"
#include "./length_collapse_policy.hpp"
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

  tf::length_collapse_policy<Real> policy(
      low, high, config.preserve_boundary, config.use_quadric,
      config.max_aspect_ratio, 1e-6, config.feature_angle,
      config.feature_weight);

  tf::points_buffer<double, Dims> old_pos;

  for (int i = 0; i < config.iterations; ++i) {
    tf::split_long_edges(he, points, frame, high, config.preserve_boundary);

    auto tagged = points.points() | tf::tag(frame);
    policy.init(he, tagged);

    if (config.parallel) {
      tf::remesh::collapse_to_exhaustion_parallel<1024>(he, tagged, policy);
    } else {
      tf::remesh::collapse_to_exhaustion(he, tagged, policy);
    }

    auto [face_im, vert_im, edge_im] = he.compact();
    points = tf::reindexed(points.points(), vert_im);

    auto deviation = tf::remesh::compute_valence_deviations(he);

    if (policy.preserve_features()) {
      // auto mask = tf::make_feature_mask(he, points.points(),
      //                                    config.feature_angle);
      policy.feature_mask().compact(edge_im, vert_im);
      tf::remesh::optimize_valence(he, deviation, policy.feature_mask(), 1);
      tf::remesh::tangential_relaxation(he, points.points(), old_pos,
                                        policy.feature_mask(),
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
