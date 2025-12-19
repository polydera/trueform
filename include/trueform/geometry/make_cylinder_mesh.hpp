/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_apply.hpp"
#include "../core/angle.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/views/sequence_range.hpp"

namespace tf {

/// @brief Creates a cylinder mesh centered at origin along the z-axis.
///
/// The cylinder extends from z = -height/2 to z = +height/2.
/// All faces are triangles with outward-facing normals (CCW winding).
///
/// @tparam Index The index type for vertices and faces.
/// @tparam RealType The floating-point type for coordinates.
/// @param segments Number of subdivisions around the circumference.
/// @param radius The radius of the cylinder.
/// @param height The height of the cylinder.
/// @return A polygons_buffer containing the cylinder mesh.
template <typename Index, typename RealType>
auto make_cylinder_mesh(Index segments, RealType radius, RealType height)
    -> tf::polygons_buffer<Index, RealType, 3, 3> {
  tf::polygons_buffer<Index, RealType, 3, 3> mesh;

  // Vertices: top_center, top_ring[segments], bottom_center, bottom_ring[segments]
  const Index num_vertices = 2 + 2 * segments;
  // Faces: top_cap[segments] + bottom_cap[segments] + side[2*segments]
  const Index num_faces = 4 * segments;

  const Index top_center = 0;
  const Index top_ring_start = 1;
  const Index bottom_center = 1 + segments;
  const Index bottom_ring_start = 2 + segments;

  const RealType half_height = height / RealType{2};

  mesh.points_buffer().allocate(num_vertices);
  mesh.faces_buffer().allocate(num_faces);

  auto& points = mesh.points_buffer();
  auto& faces = mesh.faces_buffer();

  // Center vertices
  points[top_center] = tf::point<RealType, 3>{RealType{0}, RealType{0}, half_height};
  points[bottom_center] = tf::point<RealType, 3>{RealType{0}, RealType{0}, -half_height};

  tbb::task_group tg;

  // Ring vertices
  tg.run([&] {
    tf::parallel_apply(
        tf::make_sequence_range(segments),
        [&](Index j) {
          RealType theta = RealType{360} * static_cast<RealType>(j) /
                           static_cast<RealType>(segments);
          RealType x = radius * tf::cos(tf::deg(theta));
          RealType y = radius * tf::sin(tf::deg(theta));
          points[top_ring_start + j] = tf::point<RealType, 3>{x, y, half_height};
          points[bottom_ring_start + j] = tf::point<RealType, 3>{x, y, -half_height};
        },
        tf::checked);
  });

  // Top cap faces (CCW when viewed from above)
  tg.run([&] {
    tf::parallel_apply(
        tf::make_sequence_range(segments),
        [&](Index j) {
          Index next = (j + 1) % segments;
          faces[j] = tf::make_array_like(std::array<Index, 3>{
              top_center, top_ring_start + j, top_ring_start + next});
        },
        tf::checked);
  });

  // Bottom cap faces (CCW when viewed from below)
  tg.run([&] {
    tf::parallel_apply(
        tf::make_sequence_range(segments),
        [&](Index j) {
          Index next = (j + 1) % segments;
          faces[segments + j] = tf::make_array_like(std::array<Index, 3>{
              bottom_center, bottom_ring_start + next, bottom_ring_start + j});
        },
        tf::checked);
  });

  // Side faces (two triangles per quad, CCW when viewed from outside)
  tg.run([&] {
    tf::parallel_apply(
        tf::make_sequence_range(segments),
        [&](Index j) {
          Index next = (j + 1) % segments;
          Index side_base = 2 * segments + j * 2;
          // Triangle 1: top[j] -> bottom[j] -> bottom[next]
          faces[side_base] = tf::make_array_like(std::array<Index, 3>{
              top_ring_start + j, bottom_ring_start + j, bottom_ring_start + next});
          // Triangle 2: top[j] -> bottom[next] -> top[next]
          faces[side_base + 1] = tf::make_array_like(std::array<Index, 3>{
              top_ring_start + j, bottom_ring_start + next, top_ring_start + next});
        },
        tf::checked);
  });

  tg.wait();

  return mesh;
}

} // namespace tf
