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
#include "../core/none.hpp"
#include "./collapse_checker.hpp"
#include "./collapse_edges.hpp"
#include "./collapse_handler.hpp"
#include "./decimate_config.hpp"
#include "./feature_handler.hpp"
#include "./preserve_regions.hpp"
#include "./regions/region_label.hpp"

#include <type_traits>

namespace tf::remesh {

template <typename Index, typename PointsPolicy, typename Regions>
auto decimate(tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
              tf::coordinate_type<PointsPolicy> target_proportion,
              const tf::decimate_config<tf::coordinate_type<PointsPolicy>>
                  &config,
              Regions regions)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  using Label = tf::remesh::region_label_t<Regions, Index>;
  Index target_faces = Index(he.number_of_faces() * target_proportion);

  auto score = [](const auto &he, const auto &points, auto heh,
                  const auto &handler) -> Real {
    return tf::remesh::collapse_error_quadric<Real>(
        handler._quadrics, points, he, heh, handler._config.stabilizer);
  };

  auto checker = tf::make_collapse_checker<Real>(config.min_quality, tf::none, config.check_normals);

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
  Index n_collapsed =
      tf::collapse_edges(he, points, handler, target_faces);
  return {n_collapsed, std::move(features)};
}

template <typename Index, typename PointsPolicy, typename Regions>
auto decimate(tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
              tf::coordinate_type<PointsPolicy> target_proportion,
              const tf::decimate_config<tf::coordinate_type<PointsPolicy>>
                  &config,
              Regions regions)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  return tf::remesh::decimate(he, points, target_proportion, config, regions);
}

} // namespace tf::remesh

namespace tf {

/// @ingroup remesh
/// @brief Decimate a mesh to a target face count using quadric error metrics.
template <typename Index, typename PointsPolicy>
auto decimate(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::coordinate_type<PointsPolicy> target_proportion,
    const tf::decimate_config<tf::coordinate_type<PointsPolicy>> &config =
        tf::decimate_config<tf::coordinate_type<PointsPolicy>>{}) -> Index {
  return tf::remesh::decimate(he, points, target_proportion, config, tf::none)
      .first;
}

/// @ingroup remesh
/// @brief Region-preserving decimate. Returns post-decimation face labels
/// indexed by pre-compact face ids; caller compacts `he` and remaps.
template <typename Index, typename PointsPolicy, typename Range>
auto decimate(tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
              tf::coordinate_type<PointsPolicy> target_proportion,
              const tf::decimate_config<tf::coordinate_type<PointsPolicy>>
                  &config,
              tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  using Label = typename Range::value_type;
  // Empty range carries no labels: run the non-region path, return empty labels.
  if (regions.face_regions.size() == 0) {
    tf::decimate(he, points, target_proportion, config);
    return tf::buffer<Label>{};
  }
  auto [_, features] =
      tf::remesh::decimate(he, points, target_proportion, config, regions);
  return std::move(features.face_labels);
}

template <typename Index, typename PointsPolicy>
auto decimate(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::coordinate_type<PointsPolicy> target_proportion,
    const tf::decimate_config<tf::coordinate_type<PointsPolicy>> &config)
    -> Index {
  return tf::decimate(he, points, target_proportion, config);
}

template <typename Index, typename PointsPolicy, typename Range>
auto decimate(tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
              tf::coordinate_type<PointsPolicy> target_proportion,
              const tf::decimate_config<tf::coordinate_type<PointsPolicy>>
                  &config,
              tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  return tf::decimate(he, points, target_proportion, config, regions);
}

} // namespace tf
