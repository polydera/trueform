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
#include "../core/polygons_buffer.hpp"
#include "../core/static_size.hpp"
#include "../reindex/points.hpp"
#include "../reindex/return_index_map.hpp"
#include "../topology/policy/half_edges.hpp"
#include "./decimate.hpp"
#include "./preserve_regions.hpp"

#include <utility>

namespace tf {

/// @brief Decimate and return index maps (face + vertex).
template <typename Policy>
auto decimated(
    const tf::polygons<Policy> &polygons,
    tf::coordinate_type<Policy> target_proportion,
    const tf::decimate_config<tf::coordinate_type<Policy>> &config,
    tf::return_index_map_t) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using Real = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  static_assert(tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>> ==
                3);

  if constexpr (!tf::has_half_edges_policy<Policy>) {
    tf::half_edges<Index> he(polygons);
    return decimated(polygons | tf::tag(he), target_proportion, config,
                     tf::return_index_map);
  } else {
    auto &he_view = polygons.half_edges();
    tf::half_edges<Index> he;
    auto hd = he_view.half_edges_data();
    he.half_edges_buffer().allocate(hd.size());
    tf::parallel_copy(hd, tf::make_range(he.half_edges_buffer()));
    he.rebuild_handles(he_view.n_faces(), he_view.n_vertices());

    tf::points_buffer<Real, Dims> points;
    points.allocate(polygons.points().size());
    tf::parallel_copy(polygons.points(), points.points());

    tf::decimate(he, points.points(), target_proportion, config);
    auto [fim, vim, eim] = he.compact();

    tf::polygons_buffer<Index, Real, Dims, 3> mesh;
    mesh.faces_buffer() = tf::make_faces_buffer(he);
    mesh.points_buffer() = tf::reindexed(points.points(), vim);
    return std::tuple{std::move(mesh), std::move(he), std::move(fim),
                      std::move(vim)};
  }
}

/// @brief Decimate with config.
template <typename Policy>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config) {
  auto [mesh, he, fim, vim] =
      decimated(polygons, target_proportion, config, tf::return_index_map);
  return std::pair{std::move(mesh), std::move(he)};
}

/// @brief Decimate with default config.
template <typename Policy>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion,
                   tf::decimate_config<Real>{});
}

/// @brief Decimate with default config, return index maps.
template <typename Policy>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion,
                   tf::decimate_config<Real>{}, tf::return_index_map);
}

/// @brief Region-preserving decimate. Returns (mesh, he, face_labels).
template <typename Policy, typename Range>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config,
               tf::preserve_regions_t<Range> regions) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using Real = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  static_assert(tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>> ==
                3);

  // An empty range carries no labels: run the non-region path and return an
  // empty face_labels buffer of the mesh index type. The region machinery is
  // never entered.
  if (regions.face_regions.size() == 0) {
    auto [mesh, he] = decimated(polygons, target_proportion, config);
    return std::tuple{std::move(mesh), std::move(he), tf::buffer<typename Range::value_type>{}};
  }

  if constexpr (!tf::has_half_edges_policy<Policy>) {
    tf::half_edges<Index> he(polygons);
    return decimated(polygons | tf::tag(he), target_proportion, config,
                     regions);
  } else {
    auto &he_view = polygons.half_edges();
    tf::half_edges<Index> he;
    auto hd = he_view.half_edges_data();
    he.half_edges_buffer().allocate(hd.size());
    tf::parallel_copy(hd, tf::make_range(he.half_edges_buffer()));
    he.rebuild_handles(he_view.n_faces(), he_view.n_vertices());

    tf::points_buffer<Real, Dims> points;
    points.allocate(polygons.points().size());
    tf::parallel_copy(polygons.points(), points.points());

    auto [_, features] = tf::remesh::decimate(
        he, points.points(), target_proportion, config, regions);
    auto [fim, vim, eim] = he.compact();
    features.compact(fim, eim, vim);

    tf::polygons_buffer<Index, Real, Dims, 3> mesh;
    mesh.faces_buffer() = tf::make_faces_buffer(he);
    mesh.points_buffer() = tf::reindexed(points.points(), vim);
    return std::tuple{std::move(mesh), std::move(he),
                      std::move(features.face_labels)};
  }
}

/// @brief Region-preserving decimate with default config.
template <typename Policy, typename Range>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::preserve_regions_t<Range> regions) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{},
                   regions);
}

/// @brief Region-preserving decimate with index maps. Returns
/// (mesh, he, face_im, vert_im, face_labels).
template <typename Policy, typename Range>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               const tf::decimate_config<tf::coordinate_type<Policy>> &config,
               tf::preserve_regions_t<Range> regions,
               tf::return_index_map_t) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using Real = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  static_assert(tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>> ==
                3);

  // An empty range carries no labels: run the non-region path and return an
  // empty face_labels buffer of the mesh index type. The region machinery is
  // never entered.
  if (regions.face_regions.size() == 0) {
    auto [mesh, he, fim, vim] =
        decimated(polygons, target_proportion, config, tf::return_index_map);
    return std::tuple{std::move(mesh), std::move(he), std::move(fim),
                      std::move(vim), tf::buffer<typename Range::value_type>{}};
  }

  if constexpr (!tf::has_half_edges_policy<Policy>) {
    tf::half_edges<Index> he(polygons);
    return decimated(polygons | tf::tag(he), target_proportion, config,
                     regions, tf::return_index_map);
  } else {
    auto &he_view = polygons.half_edges();
    tf::half_edges<Index> he;
    auto hd = he_view.half_edges_data();
    he.half_edges_buffer().allocate(hd.size());
    tf::parallel_copy(hd, tf::make_range(he.half_edges_buffer()));
    he.rebuild_handles(he_view.n_faces(), he_view.n_vertices());

    tf::points_buffer<Real, Dims> points;
    points.allocate(polygons.points().size());
    tf::parallel_copy(polygons.points(), points.points());

    auto [_, features] = tf::remesh::decimate(
        he, points.points(), target_proportion, config, regions);
    auto [fim, vim, eim] = he.compact();
    features.compact(fim, eim, vim);

    tf::polygons_buffer<Index, Real, Dims, 3> mesh;
    mesh.faces_buffer() = tf::make_faces_buffer(he);
    mesh.points_buffer() = tf::reindexed(points.points(), vim);
    return std::tuple{std::move(mesh), std::move(he), std::move(fim),
                      std::move(vim), std::move(features.face_labels)};
  }
}

/// @brief Region-preserving decimate with default config and index maps.
template <typename Policy, typename Range>
auto decimated(const tf::polygons<Policy> &polygons,
               tf::coordinate_type<Policy> target_proportion,
               tf::preserve_regions_t<Range> regions,
               tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return decimated(polygons, target_proportion, tf::decimate_config<Real>{},
                   regions, tf::return_index_map);
}

} // namespace tf
