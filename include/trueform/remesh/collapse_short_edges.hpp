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
#include "../core/frame.hpp"
#include "../core/frame_of.hpp"
#include "../core/none.hpp"
#include "../core/points_buffer.hpp"
#include "../core/transformed.hpp"
#include "../reindex/points.hpp"
#include "../reindex/return_index_map.hpp"
#include "./collapse_checker.hpp"
#include "./collapse_edges.hpp"
#include "./collapse_handler.hpp"
#include "./feature_handler.hpp"
#include "./length_collapse_config.hpp"
#include "./preserve_regions.hpp"
#include "./protect_vertices.hpp"
#include "./regions/region_label.hpp"

#include <tuple>
#include <utility>

namespace tf::remesh {

template <typename Index, typename PointsPolicy, typename Regions,
          typename Protection>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    Regions regions, Protection protection)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
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

  auto checker = tf::make_collapse_checker<Real>(config.min_quality, max2, config.check_normals);

  auto features = tf::remesh::build_feature_handler(
      he, points, config.feature_angle, regions, protection);

  auto handler = tf::make_collapse_handler<Real>(score, checker,
                                                  features.as_view(), config);
  Index n = tf::collapse_edges(he, points, handler);
  return {n, std::move(features)};
}

template <typename Index, typename PointsPolicy, typename Regions,
          typename Protection>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    Regions regions, Protection protection)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  return tf::remesh::collapse_short_edges(he, points, min_len, config, regions,
                                          protection);
}

template <typename Index, typename PointsPolicy, typename Regions,
          typename Protection>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    Regions regions, Protection protection)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  Real max2 = config.max_length * config.max_length;

  auto score = [](const auto &he, const auto &points, auto heh,
                   const auto &) -> Real {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    return tf::transformed(points[v1] - points[v0], tf::frame_of(points))
        .length2();
  };

  auto checker = tf::make_collapse_checker<Real>(config.min_quality, max2, config.check_normals);

  auto features = tf::remesh::build_feature_handler(
      he, points, config.feature_angle, regions, protection);

  auto handler = tf::make_collapse_handler<Real>(score, checker,
                                                  features.as_view(), config);
  Index n = tf::collapse_edges(he, points, handler, target_faces);
  return {n, std::move(features)};
}

template <typename Index, typename PointsPolicy, typename Regions,
          typename Protection>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    Regions regions, Protection protection)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  return tf::remesh::collapse_short_edges(he, points, target_faces, config,
                                          regions, protection);
}

} // namespace tf::remesh

namespace tf {

// ---------------------------------------------------------------------------
// Exhaustion mode — collapse all edges shorter than min_len
// ---------------------------------------------------------------------------

template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config = {}) -> Index {
  return tf::remesh::collapse_short_edges(he, points, min_len, config, tf::none,
                                          tf::none)
      .first;
}

template <typename Index, typename PointsPolicy, typename Range>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  using Label = typename Range::value_type;
  // Empty range carries no labels: run the non-region path, return empty labels.
  if (regions.face_regions.size() == 0) {
    tf::collapse_short_edges(he, points, min_len, config);
    return tf::buffer<Label>{};
  }
  auto [_, features] = tf::remesh::collapse_short_edges(he, points, min_len,
                                                        config, regions,
                                                        tf::none);
  return std::move(features.face_labels);
}

template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config) -> Index {
  return tf::collapse_short_edges(he, points, min_len, config);
}

template <typename Index, typename PointsPolicy, typename Range>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  return tf::collapse_short_edges(he, points, min_len, config, regions);
}

// ---------------------------------------------------------------------------
// Target mode — collapse shortest edges until target face count is reached
// ---------------------------------------------------------------------------

template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config = {}) -> Index {
  return tf::remesh::collapse_short_edges(he, points, target_faces, config,
                                          tf::none, tf::none)
      .first;
}

template <typename Index, typename PointsPolicy, typename Range>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  using Label = typename Range::value_type;
  // Empty range carries no labels: run the non-region path, return empty labels.
  if (regions.face_regions.size() == 0) {
    tf::collapse_short_edges(he, points, target_faces, config);
    return tf::buffer<Label>{};
  }
  auto [_, features] = tf::remesh::collapse_short_edges(
      he, points, target_faces, config, regions, tf::none);
  return std::move(features.face_labels);
}

template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config) -> Index {
  return tf::collapse_short_edges(he, points, target_faces, config);
}

template <typename Index, typename PointsPolicy, typename Range>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  return tf::collapse_short_edges(he, points, target_faces, config, regions);
}

// ---------------------------------------------------------------------------
// In-place driver — owned points buffer, frame-aware, compacts internally.
// Collapses every edge shorter than min_len (in the given frame), compacts the
// half-edge structure, and reindexes the points. The driver tf::collapsed_short_
// edges forwards to. Returns the extras in parameter order:
//   [face_labels][protection_mask][face_map, vertex_map]
// Length collapse is pure collapse, so both index maps are exact.
// ---------------------------------------------------------------------------

template <typename Index, typename Real, std::size_t Dims, typename FramePolicy>
auto collapse_short_edges(tf::half_edges<Index> &he,
                          tf::points_buffer<Real, Dims> &points,
                          const tf::frame_like<Dims, FramePolicy> &frame,
                          Real min_len,
                          const tf::length_collapse_config<Real> &config = {})
    -> void {
  auto tagged = points.points() | tf::tag(frame);
  tf::remesh::collapse_short_edges(he, tagged, min_len, config, tf::none,
                                   tf::none);
  auto [fim, vim, eim] = he.compact();
  points = tf::reindexed(points.points(), vim);
}

template <typename Index, typename Real, std::size_t Dims, typename FramePolicy>
auto collapse_short_edges(tf::half_edges<Index> &he,
                          tf::points_buffer<Real, Dims> &points,
                          const tf::frame_like<Dims, FramePolicy> &frame,
                          Real min_len,
                          const tf::length_collapse_config<Real> &config,
                          tf::return_index_map_t)
    -> std::pair<tf::index_map_buffer<Index>, tf::index_map_buffer<Index>> {
  auto tagged = points.points() | tf::tag(frame);
  tf::remesh::collapse_short_edges(he, tagged, min_len, config, tf::none,
                                   tf::none);
  auto [fim, vim, eim] = he.compact();
  points = tf::reindexed(points.points(), vim);
  return {std::move(fim), std::move(vim)};
}

template <typename Index, typename Real, std::size_t Dims, typename FramePolicy,
          typename Mask>
auto collapse_short_edges(tf::half_edges<Index> &he,
                          tf::points_buffer<Real, Dims> &points,
                          const tf::frame_like<Dims, FramePolicy> &frame,
                          Real min_len,
                          const tf::length_collapse_config<Real> &config,
                          tf::protect_vertices_t<Mask> protection)
    -> tf::buffer<bool> {
  auto tagged = points.points() | tf::tag(frame);
  auto [n, features] = tf::remesh::collapse_short_edges(
      he, tagged, min_len, config, tf::none, protection);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return std::move(features.protected_vertices);
}

template <typename Index, typename Real, std::size_t Dims, typename FramePolicy,
          typename Mask>
auto collapse_short_edges(tf::half_edges<Index> &he,
                          tf::points_buffer<Real, Dims> &points,
                          const tf::frame_like<Dims, FramePolicy> &frame,
                          Real min_len,
                          const tf::length_collapse_config<Real> &config,
                          tf::protect_vertices_t<Mask> protection,
                          tf::return_index_map_t)
    -> std::tuple<tf::buffer<bool>, tf::index_map_buffer<Index>,
                  tf::index_map_buffer<Index>> {
  auto tagged = points.points() | tf::tag(frame);
  auto [n, features] = tf::remesh::collapse_short_edges(
      he, tagged, min_len, config, tf::none, protection);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return {std::move(features.protected_vertices), std::move(fim),
          std::move(vim)};
}

template <typename Index, typename Real, std::size_t Dims, typename FramePolicy,
          typename Range>
auto collapse_short_edges(tf::half_edges<Index> &he,
                          tf::points_buffer<Real, Dims> &points,
                          const tf::frame_like<Dims, FramePolicy> &frame,
                          Real min_len,
                          const tf::length_collapse_config<Real> &config,
                          tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  using Label = typename Range::value_type;
  if (regions.face_regions.size() == 0) {
    tf::collapse_short_edges(he, points, frame, min_len, config);
    return tf::buffer<Label>{};
  }
  auto tagged = points.points() | tf::tag(frame);
  auto [n, features] = tf::remesh::collapse_short_edges(he, tagged, min_len,
                                                        config, regions,
                                                        tf::none);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return std::move(features.face_labels);
}

template <typename Index, typename Real, std::size_t Dims, typename FramePolicy,
          typename Range>
auto collapse_short_edges(tf::half_edges<Index> &he,
                          tf::points_buffer<Real, Dims> &points,
                          const tf::frame_like<Dims, FramePolicy> &frame,
                          Real min_len,
                          const tf::length_collapse_config<Real> &config,
                          tf::preserve_regions_t<Range> regions,
                          tf::return_index_map_t)
    -> std::tuple<tf::buffer<typename Range::value_type>,
                  tf::index_map_buffer<Index>, tf::index_map_buffer<Index>> {
  using Label = typename Range::value_type;
  if (regions.face_regions.size() == 0) {
    auto [fm, vm] =
        tf::collapse_short_edges(he, points, frame, min_len, config,
                                 tf::return_index_map);
    return {tf::buffer<Label>{}, std::move(fm), std::move(vm)};
  }
  auto tagged = points.points() | tf::tag(frame);
  auto [n, features] = tf::remesh::collapse_short_edges(he, tagged, min_len,
                                                        config, regions,
                                                        tf::none);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return {std::move(features.face_labels), std::move(fim), std::move(vim)};
}

template <typename Index, typename Real, std::size_t Dims, typename FramePolicy,
          typename Range, typename Mask>
auto collapse_short_edges(tf::half_edges<Index> &he,
                          tf::points_buffer<Real, Dims> &points,
                          const tf::frame_like<Dims, FramePolicy> &frame,
                          Real min_len,
                          const tf::length_collapse_config<Real> &config,
                          tf::preserve_regions_t<Range> regions,
                          tf::protect_vertices_t<Mask> protection)
    -> std::pair<tf::buffer<typename Range::value_type>, tf::buffer<bool>> {
  using Label = typename Range::value_type;
  if (regions.face_regions.size() == 0) {
    auto prot =
        tf::collapse_short_edges(he, points, frame, min_len, config, protection);
    return {tf::buffer<Label>{}, std::move(prot)};
  }
  auto tagged = points.points() | tf::tag(frame);
  auto [n, features] = tf::remesh::collapse_short_edges(he, tagged, min_len,
                                                        config, regions,
                                                        protection);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return {std::move(features.face_labels),
          std::move(features.protected_vertices)};
}

template <typename Index, typename Real, std::size_t Dims, typename FramePolicy,
          typename Range, typename Mask>
auto collapse_short_edges(tf::half_edges<Index> &he,
                          tf::points_buffer<Real, Dims> &points,
                          const tf::frame_like<Dims, FramePolicy> &frame,
                          Real min_len,
                          const tf::length_collapse_config<Real> &config,
                          tf::preserve_regions_t<Range> regions,
                          tf::protect_vertices_t<Mask> protection,
                          tf::return_index_map_t)
    -> std::tuple<tf::buffer<typename Range::value_type>, tf::buffer<bool>,
                  tf::index_map_buffer<Index>, tf::index_map_buffer<Index>> {
  using Label = typename Range::value_type;
  if (regions.face_regions.size() == 0) {
    auto [prot, fm, vm] = tf::collapse_short_edges(
        he, points, frame, min_len, config, protection, tf::return_index_map);
    return {tf::buffer<Label>{}, std::move(prot), std::move(fm),
            std::move(vm)};
  }
  auto tagged = points.points() | tf::tag(frame);
  auto [n, features] = tf::remesh::collapse_short_edges(he, tagged, min_len,
                                                        config, regions,
                                                        protection);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return {std::move(features.face_labels),
          std::move(features.protected_vertices), std::move(fim),
          std::move(vim)};
}

} // namespace tf
