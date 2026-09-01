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
#include "../../arrangement/mesh/materialize_mesh_triangulation.hpp"
#include "../../arrangement/mesh/mesh_triangulation.hpp"
#include "../../core/algorithm/parallel_copy_blocked.hpp"
#include "../../core/polygons.hpp"
#include "../../core/polygons_buffer.hpp"
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::geometry {

/// The mesh a triangulation states, in the index width the caller asked for.
///
/// A corner is a position in the product's own point table, so the width it is
/// written in is the caller's choice and nothing else depends on it: asking
/// for the triangulation's own width takes its product whole, and asking for
/// another writes the same corners once into that width over the same points.
template <typename OutIndex, typename Index, typename RealT, typename Int,
          std::size_t Dims, typename Faces, typename Policy>
auto make_triangulated_mesh(
    const tf::arrangement::mesh_triangulation<Index, RealT, Int, Dims, Faces>
        &triangulation,
    const tf::polygons<Policy> &polygons)
    -> tf::polygons_buffer<OutIndex, RealT, Dims, 3> {
  auto mesh =
      tf::arrangement::materialize_mesh_triangulation(triangulation, polygons);
  if constexpr (std::is_same_v<OutIndex, Index>) {
    return mesh;
  } else {
    tf::polygons_buffer<OutIndex, RealT, Dims, 3> out;
    out.faces_buffer().allocate(mesh.faces().size());
    tf::parallel_copy_blocked(mesh.faces(), out.faces());
    out.points_buffer() = std::move(mesh.points_buffer());
    return out;
  }
}

} // namespace tf::geometry
