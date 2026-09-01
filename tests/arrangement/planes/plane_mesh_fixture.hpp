/**
 * @file plane_mesh_fixture.hpp
 * @brief The plane world a MESH states, and the meshes that state one
 *
 * One face is one plane carrier, its own boundary is its constraint set, and
 * a shared mesh edge is one canonical group — which is what carries a split
 * from a face to its neighbour. The world is @ref tf::arrangement::plane_mesh_world;
 * what this states is the lattice its identities name and the flat prefix the
 * closed-world build is handed.
 *
 * The corpus is synthetic and planar per face: a convex quad, a tilted convex
 * quad, a non-convex L, a hexagon whose FIRST THREE corners are collinear, two
 * triangles sharing one edge, a bow tie, and a closed box.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "plane_arrangement_generators.hpp"

#include <trueform/arrangement/mesh/plane_mesh_world.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/static_size.hpp>
#include <trueform/exact/vertex_converter.hpp>
#include <trueform/geometry/make_box_mesh.hpp>

#include <cstddef>
#include <type_traits>

namespace tf::test {

/// A mesh's own prepared world, plus the flat prefix the closed-world build is
/// handed. The lattice the identities name is the world's own.
template <typename Int, typename Faces> struct plane_mesh_fixture {
  tf::arrangement::plane_mesh_world<plane_index_t, Int, Faces> input;
  tf::buffer<plane_index_t> vertex_offsets;
};

/// The world is a view of the mesh, so the fixture is built from it rather
/// than filled in after the fact.
template <typename Int, typename Policy>
auto make_plane_mesh_fixture(const tf::polygons<Policy> &polygons)
    -> plane_mesh_fixture<Int, std::decay_t<decltype(polygons.faces())>> {
  using Index = plane_index_t;
  using Faces = std::decay_t<decltype(polygons.faces())>;
  const auto converter = tf::exact::make_vertex_converter<Int>(polygons);
  const auto n_points = Index(polygons.points().size());
  plane_mesh_fixture<Int, Faces> fixture{
      tf::arrangement::make_plane_mesh_world<Index, Int>(
          polygons.faces(), n_points,
          [&](Index id) {
            return converter.convert(polygons.points()[std::size_t(id)]);
          }),
      tf::buffer<Index>{}};
  fixture.vertex_offsets.push_back(0);
  fixture.vertex_offsets.push_back(n_points);
  return fixture;
}

template <typename Real>
using plane_mesh_t =
    tf::polygons_buffer<plane_index_t, Real, 3, tf::dynamic_size>;

/// One convex quad in the z = 0 plane.
template <typename Real> auto make_mesh_convex_quad() -> plane_mesh_t<Real> {
  plane_mesh_t<Real> mesh;
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(3), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(3), Real(0));
  mesh.faces_buffer().push_back({0, 1, 2, 3});
  return mesh;
}

/// The same quad on a plane no axis pair projects for free.
template <typename Real> auto make_mesh_tilted_quad() -> plane_mesh_t<Real> {
  plane_mesh_t<Real> mesh;
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(0), Real(4));
  mesh.points_buffer().emplace_back(Real(4), Real(3), Real(4));
  mesh.points_buffer().emplace_back(Real(0), Real(3), Real(0));
  mesh.faces_buffer().push_back({0, 1, 2, 3});
  return mesh;
}

/// An L: a simple hexagon with one reflex corner, so no fan triangulates it.
template <typename Real> auto make_mesh_non_convex_l() -> plane_mesh_t<Real> {
  plane_mesh_t<Real> mesh;
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(6), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(6), Real(2), Real(0));
  mesh.points_buffer().emplace_back(Real(2), Real(2), Real(0));
  mesh.points_buffer().emplace_back(Real(2), Real(6), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(6), Real(0));
  mesh.faces_buffer().push_back({0, 1, 2, 3, 4, 5});
  return mesh;
}

/// A hexagon whose corners 0, 1, 2 are collinear: the frame producer must
/// scan past them or the carrier is silently a LINE.
template <typename Real> auto make_mesh_collinear_run() -> plane_mesh_t<Real> {
  plane_mesh_t<Real> mesh;
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(2), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(3), Real(0));
  mesh.points_buffer().emplace_back(Real(2), Real(4), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(3), Real(0));
  mesh.faces_buffer().push_back({0, 1, 2, 3, 4, 5});
  return mesh;
}

/// Two triangles sharing the side (0,1) on two planes of their own: one
/// canonical group carried by both carriers.
template <typename Real> auto make_mesh_shared_edge() -> plane_mesh_t<Real> {
  plane_mesh_t<Real> mesh;
  mesh.points_buffer().emplace_back(Real(-8), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(8), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(4), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(4));
  mesh.faces_buffer().push_back({0, 1, 2});
  mesh.faces_buffer().push_back({1, 0, 3});
  return mesh;
}

/// A bow tie: the loop crosses itself, so it has no triangulation on its own
/// boundary and only a resolution can state one. Its lobes are deliberately
/// unequal, so the loop's own signed area is not zero and the area law has
/// something to break on.
template <typename Real> auto make_mesh_bow_tie() -> plane_mesh_t<Real> {
  plane_mesh_t<Real> mesh;
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(3), Real(0));
  mesh.points_buffer().emplace_back(Real(6), Real(5), Real(0));
  mesh.faces_buffer().push_back({0, 1, 2, 3});
  return mesh;
}

/// A crossing face whose crossing lands ON the edge it shares with a
/// neighbour: face 0's side (0,1) is cut by its own side (2,3) at (2,0,0), and
/// face 1 — which holds the other instance of that canonical group — must
/// receive the split and triangulate into two.
template <typename Real>
auto make_mesh_split_neighbour() -> plane_mesh_t<Real> {
  plane_mesh_t<Real> mesh;
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(3), Real(1), Real(0));
  mesh.points_buffer().emplace_back(Real(1), Real(-1), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(4));
  mesh.faces_buffer().push_back({0, 1, 2, 3});
  mesh.faces_buffer().push_back({1, 0, 4});
  return mesh;
}

/// A quad stated as a pentagon: corner 4 sits exactly where corner 2 does, so
/// one side bounds nothing and the projection cannot tell the two identities
/// apart.
template <typename Real>
auto make_mesh_doubled_vertex() -> plane_mesh_t<Real> {
  plane_mesh_t<Real> mesh;
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(3), Real(0));
  mesh.points_buffer().emplace_back(Real(0), Real(3), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(3), Real(0));
  mesh.faces_buffer().push_back({0, 1, 2, 4, 3});
  return mesh;
}

/// Three collinear corners: the face bounds no area, so its carrier is a LINE
/// and its product is emptiness.
template <typename Real>
auto make_mesh_degenerate_face() -> plane_mesh_t<Real> {
  plane_mesh_t<Real> mesh;
  mesh.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(2), Real(0), Real(0));
  mesh.points_buffer().emplace_back(Real(4), Real(0), Real(0));
  mesh.faces_buffer().push_back({0, 1, 2});
  return mesh;
}

/// A closed box: every edge is shared, every face is a triangle.
template <typename Real> auto make_mesh_box() -> plane_mesh_t<Real> {
  const auto box = tf::make_box_mesh<plane_index_t>(Real(4), Real(3), Real(2));
  plane_mesh_t<Real> mesh;
  for (const auto point : box.points())
    mesh.points_buffer().emplace_back(point[0], point[1], point[2]);
  for (const auto face : box.faces())
    mesh.faces_buffer().push_back({face[0], face[1], face[2]});
  return mesh;
}

} // namespace tf::test
