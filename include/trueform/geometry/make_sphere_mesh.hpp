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
#include "../core/angle.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/views/sequence_range.hpp"

namespace tf {

/// @ingroup geometry_meshing
/// @brief Creates a UV sphere mesh centered at origin.
///
/// All faces have outward-facing normals (CCW winding).
///
/// @tparam Index The index type for vertices and faces (default: int).
/// @tparam RealType The floating-point type for coordinates.
/// @param radius The radius of the sphere.
/// @param stacks Number of horizontal subdivisions (latitude).
/// @param segments Number of vertical subdivisions (longitude).
/// @return A polygons_buffer containing the sphere mesh.
template <typename Index = int, typename RealType>
auto make_sphere_mesh(RealType radius, Index stacks, Index segments)
    -> tf::polygons_buffer<Index, RealType, 3, 3> {
  tf::polygons_buffer<Index, RealType, 3, 3> mesh;

  const Index num_middle_rings = stacks - 1;
  const Index num_vertices = 2 + num_middle_rings * segments;
  const Index num_faces = 2 * segments * num_middle_rings;
  const Index south_pole = num_vertices - 1;

  mesh.points_buffer().allocate(num_vertices);
  mesh.faces_buffer().allocate(num_faces);

  auto &points = mesh.points_buffer();
  auto &faces = mesh.faces_buffer();

  points[0] = tf::point<RealType, 3>{RealType(0), RealType(0), radius};
  points[south_pole] =
      tf::point<RealType, 3>{RealType(0), RealType(0), -radius};

  const Index top_cap_start = 0;
  const Index middle_start = segments;
  const Index bottom_cap_start = segments + 2 * (stacks - 2) * segments;
  const Index last_ring_start = 1 + (stacks - 2) * segments;

  tbb::task_group tg;

  // Middle ring vertices
  tg.run([&] {
    tf::parallel_for_each(
        tf::make_sequence_range(Index(1), stacks),
        [&](Index i) {
          RealType phi = RealType(180) * static_cast<RealType>(i) /
                         static_cast<RealType>(stacks);
          RealType z = radius * tf::cos(tf::deg(phi));
          RealType ring_radius = radius * tf::sin(tf::deg(phi));
          Index ring_start = 1 + (i - 1) * segments;

          for (Index j = 0; j < segments; ++j) {
            RealType theta = RealType(360) * static_cast<RealType>(j) /
                             static_cast<RealType>(segments);
            RealType x = ring_radius * tf::cos(tf::deg(theta));
            RealType y = ring_radius * tf::sin(tf::deg(theta));
            points[ring_start + j] = tf::point<RealType, 3>{x, y, z};
          }
        },
        tf::checked);
  });

  // Top cap faces
  tg.run([&] {
    tf::parallel_for_each(
        tf::make_sequence_range(segments),
        [&](Index j) {
          Index next = (j + 1) % segments;
          faces[top_cap_start + j] =
              tf::make_array_like(std::array<Index, 3>{0, 1 + j, 1 + next});
        },
        tf::checked);
  });

  // Middle quad faces (triangulated)
  tg.run([&] {
    tf::parallel_for_each(
        tf::make_sequence_range(stacks - 2),
        [&](Index i) {
          Index ring_start = 1 + i * segments;
          Index next_ring_start = 1 + (i + 1) * segments;
          Index face_base = middle_start + i * segments * 2;

          for (Index j = 0; j < segments; ++j) {
            Index next = (j + 1) % segments;
            Index v0 = ring_start + j;
            Index v1 = ring_start + next;
            Index v2 = next_ring_start + next;
            Index v3 = next_ring_start + j;

            faces[face_base + j * 2] =
                tf::make_array_like(std::array<Index, 3>{v0, v2, v1});
            faces[face_base + j * 2 + 1] =
                tf::make_array_like(std::array<Index, 3>{v0, v3, v2});
          }
        },
        tf::checked);
  });

  // Bottom cap faces
  tg.run([&] {
    tf::parallel_for_each(
        tf::make_sequence_range(segments),
        [&](Index j) {
          Index next = (j + 1) % segments;
          faces[bottom_cap_start + j] =
              tf::make_array_like(std::array<Index, 3>{
                  last_ring_start + j, south_pole, last_ring_start + next});
        },
        tf::checked);
  });

  tg.wait();

  return mesh;
}

} // namespace tf
