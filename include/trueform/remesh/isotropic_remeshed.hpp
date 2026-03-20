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
#include "../topology/policy/half_edges.hpp"
#include "./isotropic_remesh.hpp"

namespace tf {

template <typename Policy>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        const tf::remesh_config<tf::coordinate_type<Policy>>
                            &config) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using Real = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  static_assert(tf::static_size_v<std::decay_t<decltype(polygons.faces()[0])>> ==
                3);

  if constexpr (!tf::has_half_edges_policy<Policy>) {
    tf::half_edges<Index> he(polygons);
    return isotropic_remeshed(polygons | tf::tag(he), config);
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

    tf::isotropic_remesh(he, points, frame, config);

    tf::polygons_buffer<Index, Real, Dims, 3> mesh;
    mesh.faces_buffer() = tf::make_faces_buffer(he);
    mesh.points_buffer() = std::move(points);
    return std::pair{std::move(mesh), std::move(he)};
  }
}

template <typename Policy>
auto isotropic_remeshed(const tf::polygons<Policy> &polygons,
                        tf::coordinate_type<Policy> target_length) {
  return isotropic_remeshed(polygons,
                            tf::make_remesh_config(target_length));
}

} // namespace tf
