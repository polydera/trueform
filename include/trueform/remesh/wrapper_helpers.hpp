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
#include "../core/points_buffer.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/static_size.hpp"
#include "../topology/policy/half_edges.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::remesh {

/// @brief Build an owned half-edge structure and points buffer from polygons --
/// the shared front half of every *ed remesh wrapper, which then calls the
/// in-place driver and assembles a mesh. Copies the tagged half-edges if
/// present, else builds them.
template <typename Policy>
auto extract_he_points(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using Real = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  static_assert(
      tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>> == 3);

  tf::half_edges<Index> he;
  if constexpr (tf::has_half_edges_policy<Policy>) {
    auto &he_view = polygons.half_edges();
    auto hd = he_view.half_edges_data();
    he.half_edges_buffer().allocate(hd.size());
    tf::parallel_copy(hd, tf::make_range(he.half_edges_buffer()));
    he.rebuild_handles(he_view.n_faces(), he_view.n_vertices());
  } else {
    he = tf::half_edges<Index>(polygons);
  }
  tf::points_buffer<Real, Dims> points;
  points.allocate(polygons.points().size());
  tf::parallel_copy(polygons.points(), points.points());
  return std::pair{std::move(he), std::move(points)};
}

/// @brief Assemble an output mesh from a remeshed half-edge structure and its
/// (already reindexed) points.
template <typename Index, typename Real, std::size_t Dims>
auto make_mesh(const tf::half_edges<Index> &he,
               tf::points_buffer<Real, Dims> &&points)
    -> tf::polygons_buffer<Index, Real, Dims, 3> {
  tf::polygons_buffer<Index, Real, Dims, 3> mesh;
  mesh.faces_buffer() = tf::make_faces_buffer(he);
  mesh.points_buffer() = std::move(points);
  return mesh;
}

} // namespace tf::remesh
