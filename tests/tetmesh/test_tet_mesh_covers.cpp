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
#include <catch2/catch_test_macros.hpp>
#include <tetmesh_builders.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/tetmesh/tet_mesh_config.hpp>
#include <trueform/tetmesh/tet_mesh_status.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>

namespace {

using tc_mesh = tf::polygons_buffer<int, float, 3, 3>;

/// The volume of the cells of `tets`, in double.
template <typename Tets> auto tc_volume(const Tets &tets) -> double {
  double sum = 0;
  for (const auto &tet : tets.tetrahedra()) {
    double u[3], v[3], w[3];
    for (std::size_t k = 0; k < 3; ++k) {
      u[k] = double(tet[1][k]) - double(tet[0][k]);
      v[k] = double(tet[2][k]) - double(tet[0][k]);
      w[k] = double(tet[3][k]) - double(tet[0][k]);
    }
    sum += u[0] * (v[1] * w[2] - v[2] * w[1]) -
           u[1] * (v[0] * w[2] - v[2] * w[0]) +
           u[2] * (v[0] * w[1] - v[1] * w[0]);
  }
  return sum / 6;
}

/// Append the box from `lo` to `hi` to `mesh`, wound outward.
auto tc_add_box(tc_mesh &mesh, const std::array<float, 3> &lo,
                const std::array<float, 3> &hi) -> void {
  const int base = int(mesh.points().size());
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i)
        mesh.points_buffer().emplace_back(i ? hi[0] : lo[0], j ? hi[1] : lo[1],
                                          k ? hi[2] : lo[2]);
  const std::array<std::array<int, 4>, 6> quads{{{0, 4, 6, 2},
                                                 {1, 3, 7, 5},
                                                 {0, 1, 5, 4},
                                                 {2, 6, 7, 3},
                                                 {0, 2, 3, 1},
                                                 {4, 5, 7, 6}}};
  for (const auto &q : quads) {
    mesh.faces_buffer().emplace_back(base + q[0], base + q[1], base + q[2]);
    mesh.faces_buffer().emplace_back(base + q[0], base + q[2], base + q[3]);
  }
}

/// The box `[0, 1] x [0, 5] x [0, 1.5]` and a tetrahedron outside it that
/// touches its face `x = 1` along the one edge from `(1, 0, 1)` to
/// `(1, 3, 1)`: four faces meet at that edge, so the box's face holds it as
/// a slit, a segment the face lies on both sides of.
auto tc_slit_box() -> tc_mesh {
  tc_mesh mesh;
  tc_add_box(mesh, {0.f, 0.f, 0.f}, {1.f, 5.f, 1.5f});
  mesh.points_buffer().emplace_back(1.f, 0.f, 1.f);
  mesh.points_buffer().emplace_back(1.f, 3.f, 1.f);
  mesh.points_buffer().emplace_back(1.5f, 1.f, 1.f);
  mesh.points_buffer().emplace_back(4.f, 1.5f, 0.5f);
  mesh.faces_buffer().emplace_back(8, 10, 9);
  mesh.faces_buffer().emplace_back(8, 9, 11);
  mesh.faces_buffer().emplace_back(9, 10, 11);
  mesh.faces_buffer().emplace_back(8, 11, 10);
  return mesh;
}

} // namespace

TEST_CASE("make_tet_mesh: a face slit along another solid's edge meshes whole",
          "[tetmesh]") {
  // The slit's two rim pieces name one edge, one per side; a cover that
  // leaves both in its frontier still states each once.
  const auto [tets, report, map] = tf::test::build_tet_mesh(tc_slit_box());
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.refused_faces == 0);
  CHECK(report.census.faces.passes > 0);
  CHECK(std::abs(tc_volume(tets) - 7.625) < 1e-5);
}

TEST_CASE("make_tet_mesh: an empty surface is meshed by no cell", "[tetmesh]") {
  const auto [tets, report, map] = tf::test::build_tet_mesh(tc_mesh{});
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(tets.size() == 0);
  CHECK(map.facet_triangles.size() == 0);
  CHECK(map.face_facets.size() == 0);
}

TEST_CASE("make_tet_mesh: a face point placed before any flip splits its "
          "cell",
          "[tetmesh]") {
  // Two boxes apart: a face of one is pierced, no flip removes the edge and
  // the complex is still Delaunay when the face's point is placed. The
  // point stands beside facets its face holds, so it is placed by a split;
  // its conflict region would wrap one of them.
  tc_mesh boxes;
  tc_add_box(boxes, {1.5f, 0.f, 1.75f}, {2.375f, 1.625f, 3.75f});
  tc_add_box(boxes, {0.25f, 1.875f, 1.625f}, {2.125f, 3.5f, 2.625f});
  const auto [tets, report, map] = tf::test::build_tet_mesh(boxes);
  CHECK(report.status == tf::tet_mesh_status::meshed);
  CHECK(report.refused_faces == 0);
  CHECK(report.census.faces.points > 0);
  CHECK(std::abs(tc_volume(tets) - 5.890625) < 1e-5);
}

TEST_CASE("make_tet_mesh: the site budget is the config's",
          "[tetmesh][config]") {
  // The pierced face's point is one site past the entrance's; a budget of
  // one site per entrance site affords none.
  tc_mesh boxes;
  tc_add_box(boxes, {1.5f, 0.f, 1.75f}, {2.375f, 1.625f, 3.75f});
  tc_add_box(boxes, {0.25f, 1.875f, 1.625f}, {2.125f, 3.5f, 2.625f});
  const auto open = tf::test::build_tet_mesh_double(boxes);
  REQUIRE(std::get<1>(open).status == tf::tet_mesh_status::meshed);
  REQUIRE(std::get<1>(open).census.faces.points > 0);
  tf::tet_mesh_config tight;
  tight.site_budget_ratio = 1;
  const auto bounded = tf::test::build_tet_mesh_double(boxes, tight);
  CHECK(std::get<1>(bounded).status == tf::tet_mesh_status::refused);
  CHECK(std::get<1>(bounded).refused_faces > 0);
}
