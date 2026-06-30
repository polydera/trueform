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
#include "../core/points_buffer.hpp"
#include "../reindex/points.hpp"
#include "../reindex/return_index_map.hpp"
#include "./collapse_checker.hpp"
#include "./collapse_edges.hpp"
#include "./collapse_handler.hpp"
#include "./decimate_config.hpp"
#include "./feature_handler.hpp"
#include "./preserve_regions.hpp"
#include "./protect_vertices.hpp"
#include "./regions/region_label.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::remesh {

template <typename Index, typename PointsPolicy, typename Regions,
          typename Protection>
auto decimate(tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
              tf::coordinate_type<PointsPolicy> target_proportion,
              const tf::decimate_config<tf::coordinate_type<PointsPolicy>>
                  &config,
              Regions regions, Protection protection)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  Index target_faces = Index(he.number_of_faces() * target_proportion);

  auto score = [](const auto &he, const auto &points, auto heh,
                  const auto &handler) -> Real {
    return tf::remesh::collapse_error_quadric<Real>(
        handler._quadrics, points, he, heh, handler._config.stabilizer);
  };

  auto checker = tf::make_collapse_checker<Real>(config.min_quality, tf::none, config.check_normals);

  auto features = tf::remesh::build_feature_handler(he, points,
                                                    config.feature_angle,
                                                    regions, protection);

  auto handler = tf::make_collapse_handler<Real>(score, checker,
                                                  features.as_view(), config);
  Index n_collapsed =
      tf::collapse_edges(he, points, handler, target_faces);
  return {n_collapsed, std::move(features)};
}

template <typename Index, typename PointsPolicy, typename Regions,
          typename Protection>
auto decimate(tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
              tf::coordinate_type<PointsPolicy> target_proportion,
              const tf::decimate_config<tf::coordinate_type<PointsPolicy>>
                  &config,
              Regions regions, Protection protection)
    -> std::pair<Index,
                 feature_handler<Index, tf::remesh::region_label_t<Regions, Index>>> {
  return tf::remesh::decimate(he, points, target_proportion, config, regions,
                              protection);
}

} // namespace tf::remesh

namespace tf {

/// @ingroup remesh
/// @brief In-place decimate to a target face proportion (quadric collapse),
/// compacting the half-edge structure and reindexing the owned points buffer.
/// The driver tf::decimated forwards to; mutates he and points. Decimation is
/// pure collapse, so when a map is requested both the face and vertex index
/// maps are exact.
template <typename Index, typename Real, std::size_t Dims>
auto decimate(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              Real target_proportion,
              const tf::decimate_config<Real> &config = {}) -> void {
  tf::remesh::decimate(he, points.points(), target_proportion, config, tf::none,
                       tf::none);
  auto [fim, vim, eim] = he.compact();
  points = tf::reindexed(points.points(), vim);
}

/// @ingroup remesh
/// @brief In-place decimate returning the original->new face and vertex index
/// maps.
template <typename Index, typename Real, std::size_t Dims>
auto decimate(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              Real target_proportion, const tf::decimate_config<Real> &config,
              tf::return_index_map_t)
    -> std::pair<tf::index_map_buffer<Index>, tf::index_map_buffer<Index>> {
  tf::remesh::decimate(he, points.points(), target_proportion, config, tf::none,
                       tf::none);
  auto [fim, vim, eim] = he.compact();
  points = tf::reindexed(points.points(), vim);
  return {std::move(fim), std::move(vim)};
}

/// @ingroup remesh
/// @brief In-place decimate with pinned vertices; returns the new protection
/// mask.
template <typename Index, typename Real, std::size_t Dims, typename Mask>
auto decimate(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              Real target_proportion, const tf::decimate_config<Real> &config,
              tf::protect_vertices_t<Mask> protection) -> tf::buffer<bool> {
  auto [n, features] = tf::remesh::decimate(
      he, points.points(), target_proportion, config, tf::none, protection);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return std::move(features.protected_vertices);
}

/// @ingroup remesh
/// @brief In-place vertex-protecting decimate + the maps. Returns
/// (protection_mask, face_map, vertex_map).
template <typename Index, typename Real, std::size_t Dims, typename Mask>
auto decimate(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              Real target_proportion, const tf::decimate_config<Real> &config,
              tf::protect_vertices_t<Mask> protection, tf::return_index_map_t)
    -> std::tuple<tf::buffer<bool>, tf::index_map_buffer<Index>,
                  tf::index_map_buffer<Index>> {
  auto [n, features] = tf::remesh::decimate(
      he, points.points(), target_proportion, config, tf::none, protection);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return {std::move(features.protected_vertices), std::move(fim),
          std::move(vim)};
}

/// @ingroup remesh
/// @brief In-place region-preserving decimate; returns the post-op face labels.
template <typename Index, typename Real, std::size_t Dims, typename Range>
auto decimate(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              Real target_proportion, const tf::decimate_config<Real> &config,
              tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  using Label = typename Range::value_type;
  if (regions.face_regions.size() == 0) {
    tf::decimate(he, points, target_proportion, config);
    return tf::buffer<Label>{};
  }
  auto [n, features] = tf::remesh::decimate(
      he, points.points(), target_proportion, config, regions, tf::none);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return std::move(features.face_labels);
}

/// @ingroup remesh
/// @brief In-place region-preserving decimate + the maps. Returns
/// (face_labels, face_map, vertex_map).
template <typename Index, typename Real, std::size_t Dims, typename Range>
auto decimate(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              Real target_proportion, const tf::decimate_config<Real> &config,
              tf::preserve_regions_t<Range> regions, tf::return_index_map_t)
    -> std::tuple<tf::buffer<typename Range::value_type>,
                  tf::index_map_buffer<Index>, tf::index_map_buffer<Index>> {
  using Label = typename Range::value_type;
  if (regions.face_regions.size() == 0) {
    auto [fm, vm] =
        tf::decimate(he, points, target_proportion, config, tf::return_index_map);
    return {tf::buffer<Label>{}, std::move(fm), std::move(vm)};
  }
  auto [n, features] = tf::remesh::decimate(
      he, points.points(), target_proportion, config, regions, tf::none);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return {std::move(features.face_labels), std::move(fim), std::move(vim)};
}

/// @ingroup remesh
/// @brief In-place region-preserving, vertex-protecting decimate. Returns
/// (face_labels, protection_mask).
template <typename Index, typename Real, std::size_t Dims, typename Range,
          typename Mask>
auto decimate(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              Real target_proportion, const tf::decimate_config<Real> &config,
              tf::preserve_regions_t<Range> regions,
              tf::protect_vertices_t<Mask> protection)
    -> std::pair<tf::buffer<typename Range::value_type>, tf::buffer<bool>> {
  using Label = typename Range::value_type;
  if (regions.face_regions.size() == 0) {
    auto prot = tf::decimate(he, points, target_proportion, config, protection);
    return {tf::buffer<Label>{}, std::move(prot)};
  }
  auto [n, features] = tf::remesh::decimate(
      he, points.points(), target_proportion, config, regions, protection);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return {std::move(features.face_labels),
          std::move(features.protected_vertices)};
}

/// @ingroup remesh
/// @brief In-place region-preserving, vertex-protecting decimate + the maps.
/// Returns (face_labels, protection_mask, face_map, vertex_map).
template <typename Index, typename Real, std::size_t Dims, typename Range,
          typename Mask>
auto decimate(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              Real target_proportion, const tf::decimate_config<Real> &config,
              tf::preserve_regions_t<Range> regions,
              tf::protect_vertices_t<Mask> protection, tf::return_index_map_t)
    -> std::tuple<tf::buffer<typename Range::value_type>, tf::buffer<bool>,
                  tf::index_map_buffer<Index>, tf::index_map_buffer<Index>> {
  using Label = typename Range::value_type;
  if (regions.face_regions.size() == 0) {
    auto [prot, fm, vm] = tf::decimate(he, points, target_proportion, config,
                                       protection, tf::return_index_map);
    return {tf::buffer<Label>{}, std::move(prot), std::move(fm),
            std::move(vm)};
  }
  auto [n, features] = tf::remesh::decimate(
      he, points.points(), target_proportion, config, regions, protection);
  (void)n;
  auto [fim, vim, eim] = he.compact();
  features.compact(fim, eim, vim);
  points = tf::reindexed(points.points(), vim);
  return {std::move(features.face_labels),
          std::move(features.protected_vertices), std::move(fim),
          std::move(vim)};
}

} // namespace tf
