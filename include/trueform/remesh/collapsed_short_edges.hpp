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
#include "../core/frame_of.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/static_size.hpp"
#include "../reindex/points.hpp"
#include "../reindex/return_index_map.hpp"
#include "../topology/policy/half_edges.hpp"
#include "./collapse_short_edges.hpp"

namespace tf {

/// @brief Collapse short edges and return index maps (face + vertex).
template <typename Policy>
auto collapsed_short_edges(
    const tf::polygons<Policy> &polygons,
    tf::coordinate_type<Policy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<Policy>> &config,
    tf::return_index_map_t) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using Real = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  static_assert(tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>> ==
                3);

  if constexpr (!tf::has_half_edges_policy<Policy>) {
    tf::half_edges<Index> he(polygons);
    return collapsed_short_edges(polygons | tf::tag(he), min_len, config,
                                 tf::return_index_map);
  } else {
    auto &he_view = polygons.half_edges();
    tf::half_edges<Index> he;
    auto hd = he_view.half_edges_data();
    he.half_edges_buffer().allocate(hd.size());
    tf::parallel_copy(hd, tf::make_range(he.half_edges_buffer()));
    he.rebuild_handles(he_view.n_faces(), he_view.n_vertices());

    auto frame = tf::frame_of(polygons);

    tf::points_buffer<Real, Dims> points;
    points.allocate(polygons.points().size());
    tf::parallel_copy(polygons.points(), points.points());

    auto tagged = points.points() | tf::tag(frame);
    tf::collapse_short_edges(he, tagged, min_len, config);
    auto [fim, vim] = he.compact();

    tf::polygons_buffer<Index, Real, Dims, 3> mesh;
    mesh.faces_buffer() = tf::make_faces_buffer(he);
    mesh.points_buffer() = tf::reindexed(points.points(), vim);
    return std::tuple{std::move(mesh), std::move(he), std::move(fim),
                      std::move(vim)};
  }
}

/// @brief Collapse short edges with config.
template <typename Policy>
auto collapsed_short_edges(
    const tf::polygons<Policy> &polygons,
    tf::coordinate_type<Policy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<Policy>> &config) {
  auto [mesh, he, fim, vim] =
      collapsed_short_edges(polygons, min_len, config, tf::return_index_map);
  return std::pair{std::move(mesh), std::move(he)};
}

/// @brief Collapse short edges with default config.
template <typename Policy>
auto collapsed_short_edges(const tf::polygons<Policy> &polygons,
                           tf::coordinate_type<Policy> min_len) {
  using Real = tf::coordinate_type<Policy>;
  return collapsed_short_edges(
      polygons, min_len, tf::length_collapse_config<Real>{});
}

/// @brief Collapse short edges with default config, return index maps.
template <typename Policy>
auto collapsed_short_edges(const tf::polygons<Policy> &polygons,
                           tf::coordinate_type<Policy> min_len,
                           tf::return_index_map_t) {
  using Real = tf::coordinate_type<Policy>;
  return collapsed_short_edges(polygons, min_len,
                               tf::length_collapse_config<Real>{},
                               tf::return_index_map);
}

} // namespace tf
