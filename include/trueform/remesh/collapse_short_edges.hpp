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
#include "../core/none.hpp"
#include "../core/transformed.hpp"
#include "./collapse_checker.hpp"
#include "./collapse_edges.hpp"
#include "./collapse_handler.hpp"
#include "./feature_handler.hpp"
#include "./length_collapse_config.hpp"
#include "./preserve_regions.hpp"
#include "./regions/region_label.hpp"

#include <type_traits>
#include <utility>

namespace tf::remesh {

template <typename Index, typename PointsPolicy, typename Regions>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    Regions regions)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  using Label = tf::remesh::region_label_t<Regions, Index>;
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

  feature_handler<Index, Label> features;
  if constexpr (std::is_same_v<Regions, tf::none_t>) {
    if (config.feature_angle.value >= 0)
      features.init(he, points, config.feature_angle);
  } else {
    if (config.feature_angle.value >= 0)
      features.init(he, points, config.feature_angle, regions.face_regions);
    else
      features.init_regions(he, points, regions.face_regions);
  }

  auto handler = tf::make_collapse_handler<Real>(score, checker,
                                                  features.as_view(), config);
  Index n = tf::collapse_edges(he, points, handler);
  return {n, std::move(features)};
}

template <typename Index, typename PointsPolicy, typename Regions>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    Regions regions)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  return tf::remesh::collapse_short_edges(he, points, min_len, config, regions);
}

template <typename Index, typename PointsPolicy, typename Regions>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    Regions regions)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  using Label = tf::remesh::region_label_t<Regions, Index>;
  Real max2 = config.max_length * config.max_length;

  auto score = [](const auto &he, const auto &points, auto heh,
                   const auto &) -> Real {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    return tf::transformed(points[v1] - points[v0], tf::frame_of(points))
        .length2();
  };

  auto checker = tf::make_collapse_checker<Real>(config.min_quality, max2, config.check_normals);

  feature_handler<Index, Label> features;
  if constexpr (std::is_same_v<Regions, tf::none_t>) {
    if (config.feature_angle.value >= 0)
      features.init(he, points, config.feature_angle);
  } else {
    if (config.feature_angle.value >= 0)
      features.init(he, points, config.feature_angle, regions.face_regions);
    else
      features.init_regions(he, points, regions.face_regions);
  }

  auto handler = tf::make_collapse_handler<Real>(score, checker,
                                                  features.as_view(), config);
  Index n = tf::collapse_edges(he, points, handler, target_faces);
  return {n, std::move(features)};
}

template <typename Index, typename PointsPolicy, typename Regions>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>> &config,
    Regions regions)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  return tf::remesh::collapse_short_edges(he, points, target_faces, config,
                                          regions);
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
  return tf::remesh::collapse_short_edges(he, points, min_len,
                                                     config, tf::none)
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
  auto [_, features] = tf::remesh::collapse_short_edges(
      he, points, min_len, config, regions);
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
  return tf::remesh::collapse_short_edges(he, points, target_faces,
                                                    config, tf::none)
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
      he, points, target_faces, config, regions);
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

} // namespace tf
