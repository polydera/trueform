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
#include <catch2/catch_test_macros.hpp>
#include <tetmesh_builders.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/reindex/concatenated.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/spatial/tree_config.hpp>
#include <trueform/tetmesh/tet_mesh_config.hpp>
#include <trueform/tetmesh/tet_mesh_status.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/manifold_edge_link.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using ta_mesh = tf::polygons_buffer<int, float, 3, 3>;

/// `surface` moved by `(x, y, z)`.
auto ta_moved(ta_mesh surface, float x, float y, float z) -> ta_mesh {
  for (auto point : surface.points()) {
    point[0] += x;
    point[1] += y;
    point[2] += z;
  }
  return surface;
}

/// Two overlapping boxes: a surface the entrance must cut.
auto ta_union() -> ta_mesh {
  const auto first = tf::make_box_mesh<int>(2.f, 2.f, 2.f);
  const auto second =
      ta_moved(tf::make_box_mesh<int>(2.f, 2.f, 2.f), 0.5f, 0.75f, 1.25f);
  return tf::concatenated(first.polygons(), second.polygons());
}

/// Whether two tet meshes hold the same cells over the same points.
template <typename Tets> auto ta_same(const Tets &a, const Tets &b) -> bool {
  if (a.tetras().size() != b.tetras().size() ||
      a.points().size() != b.points().size())
    return false;
  for (std::size_t t = 0; t < a.tetras().size(); ++t)
    for (std::size_t k = 0; k < 4; ++k)
      if (a.tetras()[t][k] != b.tetras()[t][k])
        return false;
  for (std::size_t p = 0; p < a.points().size(); ++p)
    for (std::size_t k = 0; k < 3; ++k)
      if (a.points()[p][k] != b.points()[p][k])
        return false;
  return true;
}

} // namespace

TEST_CASE("make_tet_mesh: the default call emits double, the status and "
          "no index map",
          "[tetmesh][config]") {
  // The builder's declared tuple is the call's own return: a float input
  // comes back in double, the tetrahedra and the status alone; the report
  // is a stated request (tf::return_report).
  const auto box = tf::make_box_mesh<int>(1.f, 2.f, 3.f, 3, 4, 5);
  const auto plain = tf::test::build_tet_mesh_default(box);
  const auto mapped = tf::test::build_tet_mesh_double(box);
  STATIC_REQUIRE(std::tuple_size_v<std::decay_t<decltype(plain)>> == 2);
  STATIC_REQUIRE(
      std::is_same_v<std::decay_t<decltype(std::get<0>(plain).points()[0][0])>,
                     double>);
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(std::get<1>(plain))>,
                                tf::tet_mesh_status>);
  REQUIRE(std::get<1>(plain) == tf::tet_mesh_status::meshed);
  CHECK(std::get<1>(mapped).status == std::get<1>(plain));
  CHECK(ta_same(std::get<0>(plain), std::get<0>(mapped)));
}

TEST_CASE("make_tet_mesh: a tolerance of zero is the default call",
          "[tetmesh][config]") {
  const auto surface = ta_union();
  const auto plain = tf::test::build_tet_mesh_double(surface);
  const auto zero =
      tf::test::build_tet_mesh_double(surface, tf::tet_mesh_config{true, 0.0});
  REQUIRE(std::get<1>(plain).status == tf::tet_mesh_status::meshed);
  CHECK(std::get<1>(zero).status == tf::tet_mesh_status::meshed);
  CHECK(ta_same(std::get<0>(plain), std::get<0>(zero)));
}

TEST_CASE("make_tet_mesh: a tolerance reaches the entrance's arrangement",
          "[tetmesh][config]") {
  // Two unit boxes a hundred-thousandth apart: exactly, their facing faces
  // stand apart and the gap between them is no cell; under a band wider
  // than the gap the faces pool onto one plane, which the entrance arranges.
  const auto left = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  const auto right = ta_moved(tf::make_box_mesh<int>(1.f, 1.f, 1.f),
                              1.f + 1e-5f, 0.25f, 0.25f);
  const auto surface = tf::concatenated(left.polygons(), right.polygons());
  const auto exact = tf::test::build_tet_mesh_double(surface);
  const auto banded =
      tf::test::build_tet_mesh_double(surface, tf::tet_mesh_config{true, 1e-3});
  REQUIRE(std::get<1>(exact).status == tf::tet_mesh_status::meshed);
  CHECK_FALSE(ta_same(std::get<0>(exact), std::get<0>(banded)));
}

TEST_CASE("make_tet_mesh: a tolerance places the faces a drain cuts",
          "[tetmesh][config]") {
  // A unit box crossed by a taller one whose bottom stands a
  // hundred-thousandth above its own: the crossing is cut in a round either
  // way, and under a band wider than the gap the bottoms the round reaches
  // stand on one plane.
  const auto left = tf::make_box_mesh<int>(1.f, 1.f, 1.f);
  const auto right = ta_moved(tf::make_box_mesh<int>(1.f, 1.f, 2.f), 0.5f,
                              0.25f, 0.5f + 1e-5f);
  const auto surface = tf::concatenated(left.polygons(), right.polygons());
  const auto exact = tf::test::build_tet_mesh_double(surface);
  const auto banded =
      tf::test::build_tet_mesh_double(surface, tf::tet_mesh_config{true, 1e-3});
  REQUIRE(std::get<1>(exact).status == tf::tet_mesh_status::meshed);
  REQUIRE(std::get<1>(banded).status == tf::tet_mesh_status::meshed);
  CHECK(std::get<1>(banded).drains >= 1);
  CHECK_FALSE(ta_same(std::get<0>(exact), std::get<0>(banded)));
}

TEST_CASE("make_tet_mesh: a surface as given is meshed without its "
          "arrangement",
          "[tetmesh][config]") {
  // A clean surface is the same constraints either way: the same cells, and
  // no arrangement pass in the call that takes it as given.
  const auto box = tf::make_box_mesh<int>(1.f, 2.f, 3.f, 3, 4, 5);
  const auto resolved = tf::test::build_tet_mesh_double(box);
  const auto given = tf::test::build_tet_mesh_double(box, {false});
  REQUIRE(std::get<1>(resolved).status == tf::tet_mesh_status::meshed);
  CHECK(std::get<1>(resolved).drains == 0);
  CHECK(std::get<1>(resolved).pairs_named == 0);
  CHECK(std::get<1>(given).status == tf::tet_mesh_status::meshed);
  CHECK(std::get<1>(given).drains == 0);
  CHECK(std::get<1>(given).pairs_unresolved == 0);
  CHECK(ta_same(std::get<0>(resolved), std::get<0>(given)));
}

TEST_CASE("make_tet_mesh: a crossing in a surface as given refuses",
          "[tetmesh][config]") {
  const auto surface = ta_union();
  const auto resolved = tf::test::build_tet_mesh_double(surface);
  const auto given = tf::test::build_tet_mesh_double(surface, {false});
  CHECK(std::get<1>(resolved).status == tf::tet_mesh_status::meshed);
  CHECK(std::get<1>(resolved).drains >= 1);
  CHECK(std::get<1>(resolved).triangles_cut > 0);
  CHECK(std::get<1>(resolved).pairs_unresolved == 0);
  CHECK(std::get<1>(resolved).census.retired_crossing > 0);
  // every input face, cut or not, lies in a facet the mesh covers, and the
  // published facets are dense: each covers triangles
  const auto &faces_of = std::get<2>(resolved).face_facets;
  REQUIRE(std::size_t(faces_of.size()) == surface.faces().size());
  for (std::size_t f = 0; f < surface.faces().size(); ++f)
    CHECK(faces_of[f].size() >= 1);
  for (const auto &block : std::get<2>(resolved).facet_triangles)
    CHECK(block.size() >= 1);
  CHECK(std::get<1>(given).status == tf::tet_mesh_status::refused);
  CHECK(std::get<1>(given).drains == 0);
  CHECK(std::get<1>(given).pairs_unresolved > 0);
  CHECK(std::get<1>(given).pairs_unresolved == std::get<1>(given).pairs_named);
}

TEST_CASE("make_tet_mesh: a framed surface is meshed where its frame places "
          "it",
          "[tetmesh][frame]") {
  // A whole translation moves the lattice image by whole units, so the cells
  // are the unframed ones and every point is the unframed point moved, to
  // the lattice's quantum. A quarter turn moves it by a quarter turn: the
  // mesh holds the turned points and the same volume, its sites named in
  // the turned image's own order.
  const auto box = tf::make_box_mesh<int>(1.f, 2.f, 3.f, 2, 2, 2);
  auto shift = tf::make_identity_transformation<float, 3>();
  shift(0, 3) = 5.f;
  shift(1, 3) = -2.f;
  shift(2, 3) = 7.f;
  auto turn = shift;
  turn(0, 0) = 0.f;
  turn(0, 1) = -1.f;
  turn(1, 0) = 1.f;
  turn(1, 1) = 0.f;
  const auto plain = tf::test::build_tet_mesh_double(box);
  const auto moved = tf::test::build_tet_mesh_framed(box, shift);
  const auto turned = tf::test::build_tet_mesh_framed(box, turn);
  REQUIRE(std::get<1>(plain).status == tf::tet_mesh_status::meshed);
  REQUIRE(std::get<1>(moved).status == tf::tet_mesh_status::meshed);
  REQUIRE(std::get<1>(turned).status == tf::tet_mesh_status::meshed);
  const auto &a = std::get<0>(plain);
  const auto &b = std::get<0>(moved);
  REQUIRE(a.tetras().size() == b.tetras().size());
  REQUIRE(a.points().size() == b.points().size());
  bool cells = true;
  for (std::size_t t = 0; t < a.tetras().size(); ++t)
    for (std::size_t k = 0; k < 4; ++k)
      cells = cells && a.tetras()[t][k] == b.tetras()[t][k];
  CHECK(cells);
  double worst = 0;
  for (std::size_t p = 0; p < a.points().size(); ++p) {
    const auto &q = a.points()[p];
    const double to[3] = {q[0] + 5.0, q[1] - 2.0, q[2] + 7.0};
    for (std::size_t k = 0; k < 3; ++k)
      worst = std::max(worst, std::abs(b.points()[p][k] - to[k]));
  }
  CHECK(worst < 1e-6);
  const auto &c = std::get<0>(turned);
  REQUIRE(c.points().size() == a.points().size());
  std::vector<std::array<double, 3>> expected, got;
  for (const auto &q : a.points())
    expected.push_back({5.0 - q[1], q[0] - 2.0, q[2] + 7.0});
  for (const auto &q : c.points())
    got.push_back({q[0], q[1], q[2]});
  std::sort(expected.begin(), expected.end());
  std::sort(got.begin(), got.end());
  worst = 0;
  for (std::size_t p = 0; p < got.size(); ++p)
    for (std::size_t k = 0; k < 3; ++k)
      worst = std::max(worst, std::abs(got[p][k] - expected[p][k]));
  CHECK(worst < 1e-6);
  const auto volume = [](const auto &tets) {
    double sum = 0;
    for (const auto &tet : tets.tetrahedra()) {
      double u[3], v[3], w[3];
      for (std::size_t k = 0; k < 3; ++k) {
        u[k] = tet[1][k] - tet[0][k];
        v[k] = tet[2][k] - tet[0][k];
        w[k] = tet[3][k] - tet[0][k];
      }
      sum += (u[0] * (v[1] * w[2] - v[2] * w[1]) -
              u[1] * (v[0] * w[2] - v[2] * w[0]) +
              u[2] * (v[0] * w[1] - v[1] * w[0])) /
             6;
    }
    return sum;
  };
  CHECK(std::abs(volume(c) - 6.0) < 1e-9);
  CHECK(std::get<1>(turned).inverted == 0);
  CHECK(std::get<1>(turned).flat == 0);
  CHECK(std::get<1>(turned).negative_cells == 0);
  CHECK(std::abs(volume(a) - 6.0) < 1e-9);
}

TEST_CASE("make_tet_mesh: a reflecting frame flips the winding the region "
          "reads",
          "[tetmesh][frame]") {
  // Winding is semantic: mirrored, the box winds inward, so the call meshes
  // no cell and counts the cells it would have kept.
  const auto box = tf::make_box_mesh<int>(1.f, 2.f, 3.f, 2, 2, 2);
  auto mirror = tf::make_identity_transformation<float, 3>();
  mirror(0, 0) = -1.f;
  const auto mirrored = tf::test::build_tet_mesh_framed(box, mirror);
  CHECK(std::get<1>(mirrored).status == tf::tet_mesh_status::meshed);
  CHECK(std::get<0>(mirrored).tetras().size() == 0);
  CHECK(std::get<1>(mirrored).negative_cells > 0);
}

TEST_CASE("make_tet_mesh: a tagged membership and link are the ones read",
          "[tetmesh][config]") {
  // Tagged with its own structures the box meshes to the untagged cells; a
  // tagged link whose peers all read boundary is read as such: no two
  // triangles join, so every triangle is a facet of its own, and the
  // recovered surface those facets make is closed, so the box still meshes.
  const auto box = tf::make_box_mesh<int>(1.f, 2.f, 3.f, 2, 2, 2);
  tf::face_membership<int> membership(box.polygons());
  tf::manifold_edge_link<int, 3> link(box.faces(), membership);
  const auto plain = tf::test::build_tet_mesh_double(box);
  const auto tagged = tf::test::build_tet_mesh_tagged(box, membership, link);
  REQUIRE(std::get<1>(plain).status == tf::tet_mesh_status::meshed);
  CHECK(std::get<1>(tagged).status == tf::tet_mesh_status::meshed);
  CHECK(ta_same(std::get<0>(plain), std::get<0>(tagged)));
  auto broken = link;
  for (std::size_t f = 0; f < std::size_t(broken.size()); ++f)
    for (std::size_t k = 0; k < 3; ++k)
      broken[f][k] = {-1};
  const auto read = tf::test::build_tet_mesh_tagged(box, membership, broken);
  CHECK(std::get<1>(read).status == tf::tet_mesh_status::meshed);
  CHECK(std::get<1>(read).recovered_open_edges == 0);
  CHECK(std::size_t(std::get<2>(read).facet_triangles.size()) ==
        box.faces().size());
  CHECK(std::get<2>(plain).facet_triangles.size() <
        std::get<2>(read).facet_triangles.size());
  // A link tagged without its membership: the membership is built for the
  // call and the box meshes to the untagged cells.
  const auto link_only = tf::test::build_tet_mesh_with_link(box, link);
  CHECK(std::get<1>(link_only).status == tf::tet_mesh_status::meshed);
  CHECK(ta_same(std::get<0>(plain), std::get<0>(link_only)));
}

TEST_CASE("make_tet_mesh: the config converts from a bool, never from a "
          "number, and its ratio is an integer set by name",
          "[tetmesh][config]") {
  STATIC_REQUIRE(std::is_convertible_v<bool, tf::tet_mesh_config>);
  STATIC_REQUIRE_FALSE(std::is_convertible_v<double, tf::tet_mesh_config>);
  STATIC_REQUIRE_FALSE(std::is_constructible_v<tf::tet_mesh_config, double>);
  STATIC_REQUIRE_FALSE(std::is_constructible_v<tf::tet_mesh_config, int>);
  STATIC_REQUIRE(
      std::is_integral_v<decltype(tf::tet_mesh_config::site_budget_ratio)>);
  constexpr tf::tet_mesh_config given{false};
  STATIC_REQUIRE(!given.split_constraints);
  constexpr tf::tet_mesh_config banded{true, 1e-3};
  STATIC_REQUIRE(banded.split_constraints);
  STATIC_REQUIRE(banded.tolerance == 1e-3);
  tf::tet_mesh_config stated;
  stated.site_budget_ratio = 8;
  CHECK(stated.site_budget_ratio == 8);
}

TEST_CASE("make_tet_mesh: a tagged tree meshes as the untagged call",
          "[tetmesh][config]") {
  // Two overlapping boxes cross each other; the recovery finds the crossings
  // and no tree of the input enters the call, so the mesh is the same.
  const auto surface = ta_union();
  tf::aabb_tree<int, float, 3> own;
  own.build(surface.polygons(), tf::config_tree(4, 12));
  const auto plain = tf::test::build_tet_mesh_double(surface);
  const auto walked = tf::test::build_tet_mesh_with_tree(surface, own);
  REQUIRE(std::get<1>(plain).status == tf::tet_mesh_status::meshed);
  CHECK(ta_same(std::get<0>(plain), std::get<0>(walked)));
}
