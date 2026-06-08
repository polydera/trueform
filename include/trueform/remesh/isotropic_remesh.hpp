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
#include "../core/none.hpp"
#include "../core/points_buffer.hpp"
#include "../reindex/points.hpp"
#include "./collapse_checker.hpp"
#include "./collapse_edges.hpp"
#include "./collapse_handler.hpp"
#include "./feature_handler.hpp"
#include "./improve_triangulation.hpp"
#include "./optimize_valence.hpp"
#include "./preserve_regions.hpp"
#include "./regions/region_label.hpp"
#include "./isotropic_remesh_config.hpp"
#include "./split_edges.hpp"
#include "./split_handler.hpp"
#include "./tangential_relaxation.hpp"
#include "./valence_deviation.hpp"

#include <type_traits>
#include <utility>

namespace tf::remesh {

/// @ingroup remesh
/// @brief Generic isotropic remesh templated on Regions (tf::none_t or
/// tf::preserve_regions_t<...>). Returns the post-remesh feature_handler.
template <typename Index, typename Real, std::size_t Dims,
          typename FramePolicy, typename Regions>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points,
                      const tf::frame_like<Dims, FramePolicy> &frame,
                      const tf::isotropic_remesh_config<Real> &config,
                      Regions regions)
    -> feature_handler<Index, tf::remesh::region_label_t<Regions, Index>> {
  using Label = tf::remesh::region_label_t<Regions, Index>;
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
  auto checker = tf::make_collapse_checker<Real>(config.min_quality, high2, config.check_normals);

  feature_handler<Index, Label> features;
  if constexpr (std::is_same_v<Regions, tf::none_t>) {
    if (config.feature_angle.value >= 0)
      features.init(he, points.points(), config.feature_angle);
  } else {
    if (config.feature_angle.value >= 0)
      features.init(he, points.points(), config.feature_angle,
                    regions.face_regions);
    else
      features.init_regions(he, points.points(), regions.face_regions);
  }

  Real high2_split = high * high;
  auto split_handler = tf::make_split_handler<Real>(
      [high2_split, &frame](const auto &he, const auto &points,
                            auto heh) -> Real {
        auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
        auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
        auto len2 =
            tf::transformed(points[v1] - points[v0], frame).length2();
        return len2 / high2_split;
      },
      features.as_view(), config.preserve_boundary);

  auto collapse_handler = tf::make_collapse_handler<Real>(
      collapse_score, checker, features.as_view(), config);

  tf::points_buffer<double, Dims> old_pos;

  for (int i = 0; i < config.iterations; ++i) {
    tf::split_edges(he, points, frame, split_handler);
    {
      auto tagged = points.points() | tf::tag(frame);
      tf::collapse_edges(he, tagged, collapse_handler);
    }

    auto [face_im, vert_im, edge_im] = he.compact();
    points = tf::reindexed(points.points(), vert_im);

    // One {flip, relax} round; the outer loop provides the repetition. old_pos
    // is reused across all iterations.
    tf::improve_config<Real> icfg{1, config.relaxation_iters, config.lambda,
                                  config.check_normals,
                                  tf::flip_objective::valence};
    if (!features.empty()) {
      features.compact(face_im, edge_im, vert_im);
      features.recompute(he, points.points(), config.feature_angle);
      tf::remesh::improve_triangulation(he, points.points(), features.mask,
                                        old_pos, icfg);
    } else {
      tf::remesh::improve_triangulation(he, points.points(), tf::none, old_pos,
                                        icfg);
    }
  }
  return features;
}

} // namespace tf::remesh

namespace tf {

/// @ingroup remesh
/// @brief Standard isotropic remeshing (frame-aware).
template <typename Index, typename Real, std::size_t Dims, typename FramePolicy>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points,
                      const tf::frame_like<Dims, FramePolicy> &frame,
                      const tf::isotropic_remesh_config<Real> &config) -> void {
  tf::remesh::isotropic_remesh(he, points, frame, config, tf::none);
}

/// @ingroup remesh
/// @brief Region-preserving isotropic remeshing (frame-aware). Returns the
/// post-remesh per-face region labels.
template <typename Index, typename Real, std::size_t Dims,
          typename FramePolicy, typename Range>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points,
                      const tf::frame_like<Dims, FramePolicy> &frame,
                      const tf::isotropic_remesh_config<Real> &config,
                      tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  using Label = typename Range::value_type;
  // Empty range carries no labels: run the non-region path, return empty labels.
  if (regions.face_regions.size() == 0) {
    tf::isotropic_remesh(he, points, frame, config);
    return tf::buffer<Label>{};
  }
  auto features =
      tf::remesh::isotropic_remesh(he, points, frame, config, regions);
  return std::move(features.face_labels);
}

/// @ingroup remesh
/// @brief Standard isotropic remeshing.
/// @overload
template <typename Index, typename Real, std::size_t Dims>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points,
                      const tf::isotropic_remesh_config<Real> &config) -> void {
  tf::isotropic_remesh(he, points, tf::identity_frame<Real, Dims>{}, config);
}

/// @ingroup remesh
/// @brief Region-preserving isotropic remeshing.
/// @overload
template <typename Index, typename Real, std::size_t Dims, typename Range>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points,
                      const tf::isotropic_remesh_config<Real> &config,
                      tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  return tf::isotropic_remesh(he, points, tf::identity_frame<Real, Dims>{},
                              config, regions);
}

/// @brief Overload with just target edge length.
template <typename Index, typename Real, std::size_t Dims>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points, Real target_length)
    -> void {
  tf::isotropic_remesh(he, points, tf::make_isotropic_remesh_config(target_length));
}

/// @brief Region-preserving overload with just target edge length.
template <typename Index, typename Real, std::size_t Dims, typename Range>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      tf::points_buffer<Real, Dims> &points, Real target_length,
                      tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  return tf::isotropic_remesh(he, points,
                              tf::make_isotropic_remesh_config(target_length), regions);
}

/// @brief Overload taking const points, returns a new points buffer.
template <typename Index, typename PointsPolicy>
auto isotropic_remesh(
    tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    const tf::isotropic_remesh_config<tf::coordinate_type<PointsPolicy>> &config)
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

/// @brief Region-preserving overload taking const points; returns the new
/// points buffer paired with the post-remesh face labels.
template <typename Index, typename PointsPolicy, typename Range>
auto isotropic_remesh(
    tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    const tf::isotropic_remesh_config<tf::coordinate_type<PointsPolicy>> &config,
    tf::preserve_regions_t<Range> regions)
    -> std::pair<tf::points_buffer<tf::coordinate_type<PointsPolicy>,
                                   tf::coordinate_dims_v<PointsPolicy>>,
                 tf::buffer<typename Range::value_type>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<PointsPolicy>;
  auto frame = tf::frame_of(points);
  tf::points_buffer<Real, Dims> buf;
  buf.allocate(points.size());
  tf::parallel_copy(points, buf.points());
  auto labels = tf::isotropic_remesh(he, buf, frame, config, regions);
  return {std::move(buf), std::move(labels)};
}

/// @brief Overload taking const points with just target length.
template <typename Index, typename PointsPolicy>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      const tf::points<PointsPolicy> &points,
                      tf::coordinate_type<PointsPolicy> target_length)
    -> tf::points_buffer<tf::coordinate_type<PointsPolicy>,
                         tf::coordinate_dims_v<PointsPolicy>> {
  return tf::isotropic_remesh(he, points,
                              tf::make_isotropic_remesh_config(target_length));
}

/// @brief Region-preserving overload taking const points with just target
/// length.
template <typename Index, typename PointsPolicy, typename Range>
auto isotropic_remesh(tf::half_edges<Index> &he,
                      const tf::points<PointsPolicy> &points,
                      tf::coordinate_type<PointsPolicy> target_length,
                      tf::preserve_regions_t<Range> regions)
    -> std::pair<tf::points_buffer<tf::coordinate_type<PointsPolicy>,
                                   tf::coordinate_dims_v<PointsPolicy>>,
                 tf::buffer<typename Range::value_type>> {
  return tf::isotropic_remesh(
      he, points, tf::make_isotropic_remesh_config(target_length), regions);
}

} // namespace tf
