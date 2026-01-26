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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/views/sequence_range.hpp"

namespace tf {

/// @ingroup geometry_meshing
/// @brief Creates a flat rectangular plane mesh in the XY plane, centered at origin.
///
/// The plane lies at z = 0 with normal pointing +z (CCW winding).
///
/// @tparam Index The index type for vertices and faces (default: int).
/// @tparam RealType The floating-point type for coordinates.
/// @param width The size along the x-axis.
/// @param height The size along the y-axis.
/// @param width_ticks Number of subdivisions along x-axis.
/// @param height_ticks Number of subdivisions along y-axis.
/// @return A polygons_buffer containing the plane mesh.
template <typename Index = int, typename RealType>
auto make_plane_mesh(RealType width, RealType height, Index width_ticks,
                     Index height_ticks)
    -> tf::polygons_buffer<Index, RealType, 3, 3> {
  tf::polygons_buffer<Index, RealType, 3, 3> mesh;

  const Index num_vertices = (width_ticks + 1) * (height_ticks + 1);
  const Index num_faces = width_ticks * height_ticks * 2;

  mesh.points_buffer().allocate(num_vertices);
  mesh.faces_buffer().allocate(num_faces);

  auto& points = mesh.points_buffer();
  auto& faces = mesh.faces_buffer();

  const RealType hw = width / RealType{2};
  const RealType hh = height / RealType{2};
  const RealType dx = width / static_cast<RealType>(width_ticks);
  const RealType dy = height / static_cast<RealType>(height_ticks);

  // Generate vertices
  tf::parallel_for_each(
      tf::make_sequence_range(height_ticks + 1),
      [&](Index j) {
        RealType y = -hh + static_cast<RealType>(j) * dy;
        for (Index i = 0; i <= width_ticks; ++i) {
          RealType x = -hw + static_cast<RealType>(i) * dx;
          points[j * (width_ticks + 1) + i] =
              tf::point<RealType, 3>{x, y, RealType{0}};
        }
      },
      tf::checked);

  // Generate faces (2 triangles per cell, CCW for +z normal)
  tf::parallel_for_each(
      tf::make_sequence_range(height_ticks),
      [&](Index j) {
        for (Index i = 0; i < width_ticks; ++i) {
          Index v0 = j * (width_ticks + 1) + i;
          Index v1 = v0 + 1;
          Index v2 = v0 + (width_ticks + 1);
          Index v3 = v2 + 1;

          Index face_base = (j * width_ticks + i) * 2;
          // Triangle 1: v0 -> v1 -> v3
          faces[face_base] = std::array<Index, 3>{v0, v1, v3};
          // Triangle 2: v0 -> v3 -> v2
          faces[face_base + 1] = std::array<Index, 3>{v0, v3, v2};
        }
      },
      tf::checked);

  return mesh;
}

/// @ingroup geometry_meshing
/// @brief Creates a flat rectangular plane mesh in the XY plane, centered at origin.
///
/// The plane lies at z = 0 with normal pointing +z (CCW winding).
///
/// @tparam Index The index type for vertices and faces (default: int).
/// @tparam RealType The floating-point type for coordinates.
/// @param width The size along the x-axis.
/// @param height The size along the y-axis.
/// @return A polygons_buffer containing the plane mesh (4 vertices, 2 triangles).
template <typename Index = int, typename RealType>
auto make_plane_mesh(RealType width, RealType height)
    -> tf::polygons_buffer<Index, RealType, 3, 3> {
  return make_plane_mesh<Index>(width, height, Index{1}, Index{1});
}

} // namespace tf
