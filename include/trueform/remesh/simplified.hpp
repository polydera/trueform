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
#include "../topology/policy/half_edges.hpp"
#include "./preserve_regions.hpp"
#include "./simplify.hpp"

#include <utility>

namespace tf {

// NOTE: simplify runs an edge-flip + relaxation cleanup (optimize_iterations,
// default > 0), so it does NOT return index maps -- a flip rewrites face
// connectivity, making an original->final face map meaningless. Use
// tf::preserve_regions to carry per-face labels through instead. (Pure-collapse
// decimation, tf::decimated, is where index maps live.)

/// @brief Simplify to an error budget. Returns (mesh, half_edges).
template <typename Policy>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using Real = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  static_assert(
      tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>> == 3);

  if constexpr (!tf::has_half_edges_policy<Policy>) {
    tf::half_edges<Index> he(polygons);
    return simplified(polygons | tf::tag(he), config);
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

    auto features = tf::remesh::simplify(he, points, config, tf::none);
    (void)features;
    tf::polygons_buffer<Index, Real, Dims, 3> mesh;
    mesh.faces_buffer() = tf::make_faces_buffer(he);
    mesh.points_buffer() = std::move(points);
    return std::pair{std::move(mesh), std::move(he)};
  }
}

/// @brief Simplify with default config.
template <typename Policy>
auto simplified(const tf::polygons<Policy> &polygons) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{});
}

/// @brief Region-preserving simplify. Returns (mesh, he, face_labels).
template <typename Policy, typename Range>
auto simplified(const tf::polygons<Policy> &polygons,
                const tf::simplify_config<tf::coordinate_type<Policy>> &config,
                tf::preserve_regions_t<Range> regions) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using Real = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  static_assert(
      tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>> == 3);

  // An empty range carries no labels: run the non-region path and return an
  // empty face_labels buffer of the mesh index type. The region machinery is
  // never entered.
  if (regions.face_regions.size() == 0) {
    auto [mesh, he] = simplified(polygons, config);
    return std::tuple{std::move(mesh), std::move(he), tf::buffer<typename Range::value_type>{}};
  }

  if constexpr (!tf::has_half_edges_policy<Policy>) {
    tf::half_edges<Index> he(polygons);
    return simplified(polygons | tf::tag(he), config, regions);
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

    auto features = tf::remesh::simplify(he, points, config, regions);
    tf::polygons_buffer<Index, Real, Dims, 3> mesh;
    mesh.faces_buffer() = tf::make_faces_buffer(he);
    mesh.points_buffer() = std::move(points);
    return std::tuple{std::move(mesh), std::move(he),
                      std::move(features.face_labels)};
  }
}

/// @brief Region-preserving simplify with default config.
template <typename Policy, typename Range>
auto simplified(const tf::polygons<Policy> &polygons,
                tf::preserve_regions_t<Range> regions) {
  using Real = tf::coordinate_type<Policy>;
  return simplified(polygons, tf::simplify_config<Real>{}, regions);
}

} // namespace tf
