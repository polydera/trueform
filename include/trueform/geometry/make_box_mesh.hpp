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
#include "../clean/soup/polygons.hpp"
#include "../core/polygons_buffer.hpp"

namespace tf {

/// @ingroup geometry_meshing
/// @brief Creates an axis-aligned box mesh centered at origin.
///
/// All faces have outward-facing normals (CCW winding).
///
/// @tparam Index The index type for vertices and faces (default: int).
/// @tparam RealType The floating-point type for coordinates.
/// @param width The size along the x-axis.
/// @param height The size along the y-axis.
/// @param depth The size along the z-axis.
/// @return A polygons_buffer containing the box mesh (12 triangles).
template <typename Index = int, typename RealType>
auto make_box_mesh(RealType width, RealType height, RealType depth)
    -> tf::polygons_buffer<Index, RealType, 3, 3> {
  tf::polygons_buffer<Index, RealType, 3, 3> mesh;

  mesh.points_buffer().allocate(8);
  mesh.faces_buffer().allocate(12);

  auto& points = mesh.points_buffer();
  auto& faces = mesh.faces_buffer();

  const RealType hw = width / RealType{2};
  const RealType hh = height / RealType{2};
  const RealType hd = depth / RealType{2};

  // 8 corner vertices
  points[0] = tf::point<RealType, 3>{-hw, -hh, -hd};
  points[1] = tf::point<RealType, 3>{+hw, -hh, -hd};
  points[2] = tf::point<RealType, 3>{-hw, +hh, -hd};
  points[3] = tf::point<RealType, 3>{+hw, +hh, -hd};
  points[4] = tf::point<RealType, 3>{-hw, -hh, +hd};
  points[5] = tf::point<RealType, 3>{+hw, -hh, +hd};
  points[6] = tf::point<RealType, 3>{-hw, +hh, +hd};
  points[7] = tf::point<RealType, 3>{+hw, +hh, +hd};

  // 12 triangles (2 per face, CCW for outward normals)
  // Front (+z)
  faces[0] = std::array{4, 5, 7};
  faces[1] = std::array{4, 7, 6};
  // Back (-z)
  faces[2] = std::array{1, 0, 2};
  faces[3] = std::array{1, 2, 3};
  // Right (+x)
  faces[4] = std::array{5, 1, 3};
  faces[5] = std::array{5, 3, 7};
  // Left (-x)
  faces[6] = std::array{0, 4, 6};
  faces[7] = std::array{0, 6, 2};
  // Top (+y)
  faces[8] = std::array{2, 6, 7};
  faces[9] = std::array{2, 7, 3};
  // Bottom (-y)
  faces[10] = std::array{0, 1, 5};
  faces[11] = std::array{0, 5, 4};

  return mesh;
}

/// @ingroup geometry_meshing
/// @brief Creates a subdivided axis-aligned box mesh centered at origin.
///
/// All faces have outward-facing normals (CCW winding).
/// Vertices are deduplicated at edges and corners.
///
/// @tparam Index The index type for vertices and faces (default: int).
/// @tparam RealType The floating-point type for coordinates.
/// @param width The size along the x-axis.
/// @param height The size along the y-axis.
/// @param depth The size along the z-axis.
/// @param width_ticks Number of subdivisions along x-axis.
/// @param height_ticks Number of subdivisions along y-axis.
/// @param depth_ticks Number of subdivisions along z-axis.
/// @return A polygons_buffer containing the subdivided box mesh.
template <typename Index = int, typename RealType>
auto make_box_mesh(RealType width, RealType height, RealType depth,
                   Index width_ticks, Index height_ticks, Index depth_ticks)
    -> tf::polygons_buffer<Index, RealType, 3, 3> {
  const Index wt = width_ticks;
  const Index ht = height_ticks;
  const Index dt = depth_ticks;

  // Triangles per face: 2 * u_ticks * v_ticks
  const Index num_tris = 2 * (wt * ht + wt * ht + dt * ht + dt * ht +
                              wt * dt + wt * dt);

  // Soup buffer: 3 vertices per triangle, 3 coords per vertex
  tf::buffer<RealType> soup;
  soup.allocate(num_tris * 9);

  const RealType hw = width / RealType{2};
  const RealType hh = height / RealType{2};
  const RealType hd = depth / RealType{2};
  const RealType dx = width / static_cast<RealType>(wt);
  const RealType dy = height / static_cast<RealType>(ht);
  const RealType dz = depth / static_cast<RealType>(dt);

  Index offset = 0;

  // Canonical coordinate function - ensures identical FP results for same (i,j,k)
  auto coord = [&](Index i, Index j, Index k) {
    return tf::point<RealType, 3>{
        -hw + static_cast<RealType>(i) * dx,
        -hh + static_cast<RealType>(j) * dy,
        -hd + static_cast<RealType>(k) * dz};
  };

  // Helper to emit a triangle (3 vertices, 9 floats)
  auto emit_tri = [&](tf::point<RealType, 3> a, tf::point<RealType, 3> b,
                      tf::point<RealType, 3> c) {
    soup[offset++] = a[0]; soup[offset++] = a[1]; soup[offset++] = a[2];
    soup[offset++] = b[0]; soup[offset++] = b[1]; soup[offset++] = b[2];
    soup[offset++] = c[0]; soup[offset++] = c[1]; soup[offset++] = c[2];
  };

  // Helper to generate a subdivided face as triangle soup
  auto gen_face = [&](auto get_point, Index u_ticks, Index v_ticks) {
    for (Index v = 0; v < v_ticks; ++v) {
      for (Index u = 0; u < u_ticks; ++u) {
        auto p00 = get_point(u, v);
        auto p10 = get_point(u + 1, v);
        auto p01 = get_point(u, v + 1);
        auto p11 = get_point(u + 1, v + 1);
        // Two triangles per quad (CCW)
        emit_tri(p00, p10, p11);
        emit_tri(p00, p11, p01);
      }
    }
  };

  // Front (+z, k=dt): i=u, j=v
  gen_face([&](Index u, Index v) { return coord(u, v, dt); }, wt, ht);

  // Back (-z, k=0): i=wt-u (reversed for CCW), j=v
  gen_face([&](Index u, Index v) { return coord(wt - u, v, 0); }, wt, ht);

  // Right (+x, i=wt): j=v, k=dt-u (reversed for CCW)
  gen_face([&](Index u, Index v) { return coord(wt, v, dt - u); }, dt, ht);

  // Left (-x, i=0): j=v, k=u
  gen_face([&](Index u, Index v) { return coord(0, v, u); }, dt, ht);

  // Top (+y, j=ht): i=u, k=dt-v (reversed for CCW)
  gen_face([&](Index u, Index v) { return coord(u, ht, dt - v); }, wt, dt);

  // Bottom (-y, j=0): i=u, k=v
  gen_face([&](Index u, Index v) { return coord(u, 0, v); }, wt, dt);

  // Deduplicate via polygon_soup
  tf::clean::polygon_soup<Index, RealType, 3, 3> cleaned;
  cleaned.build(std::move(soup));
  return cleaned;
}

} // namespace tf
