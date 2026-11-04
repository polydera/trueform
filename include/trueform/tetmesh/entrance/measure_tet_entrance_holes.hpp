/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/buffer.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/plane_face_support.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf::tetmesh {

/// The holes an arrangement pass of the entrance leaves: `{planes, faces}`,
/// the carriers `graph.failed()` names that a face of `polygons` bounding
/// area stands on, and those faces. A failed carrier holds no triangles, so
/// such a face is missing from the product; a carrier only collapsed faces
/// stand on bounds no area and holds its product by emitting nothing.
template <typename Graph, typename Polygons>
auto measure_tet_entrance_holes(const Graph &graph, const Polygons &polygons)
    -> std::array<std::size_t, 2> {
  using Int = tf::coordinate_type<decltype(polygons.points()[0])>;
  using Index = std::decay_t<decltype(graph.arrangement().n_planes())>;
  const auto failed = graph.failed();
  if (!failed.size())
    return {0, 0};
  const auto &world = graph.world();
  const auto &arrangement = graph.arrangement();
  tf::buffer<signed char> state;
  state.allocate(std::size_t(arrangement.n_planes()));
  std::fill(state.begin(), state.end(), static_cast<signed char>(0));
  for (auto plane : failed)
    state[std::size_t(plane)] = 1;
  std::size_t faces = 0;
  const auto visit = [&](Index plane, std::size_t object) {
    if (!state[std::size_t(plane)])
      return;
    const auto &face = polygons.faces()[object];
    const auto support = tf::intersect::graph::plane_face_support<Int>(
        face, [&polygons, &face](std::size_t corner) {
          const auto &p = polygons.points()[std::size_t(face[corner])];
          return tf::exact::pt3<Int>{Int(p[0]), Int(p[1]), Int(p[2])};
        });
    if (support.size < 3)
      return;
    state[std::size_t(plane)] = 2;
    ++faces;
  };
  for (Index face = 0; face < arrangement.n_base_faces(); ++face)
    visit(world.plane_of_face(face),
          std::size_t(world.descriptor(face).object));
  const auto planes_of = arrangement.promoted_plane_of();
  const auto descriptors = arrangement.promoted_descriptors();
  for (std::size_t i = 0; i < planes_of.size(); ++i)
    visit(planes_of[i], std::size_t(descriptors[i].object));
  return {std::size_t(std::count(state.begin(), state.end(),
                                 static_cast<signed char>(2))),
          faces};
}

} // namespace tf::tetmesh
