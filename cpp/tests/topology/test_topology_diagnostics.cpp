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
#include "carriers.hpp"
#include "fixtures.hpp"

#include "trueform/cpp/topology/async/mesh_queries.hpp"
#include "trueform/cpp/topology/boundary_rims.hpp"
#include "trueform/cpp/topology/is_manifold.hpp"
#include "trueform/cpp/topology/non_manifold_vertices.hpp"
#include "trueform/cpp/topology/split_non_manifold_vertices.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <future>
#include <type_traits>
#include <vector>

namespace {

/// Two triangles meeting at vertex 0 alone.
template <typename Index, typename Real>
auto diagnostics_bowtie() -> tf::cpp::test::owned_mesh<Index, Real> {
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 1, 2, 0, 3, 4},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}, Real{-1}, Real{0}, Real{0}, Real{0}, Real{-1}, Real{0}})};
}

/// The same bowtie, stored at mixed arity.
template <typename Index, typename Real>
auto diagnostics_mixed_bowtie()
    -> tf::cpp::test::owned_mesh<Index, Real, 3, tf::dynamic_size> {
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 3, 6}, {0, 1, 2, 0, 3, 4},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}, Real{-1}, Real{0}, Real{0}, Real{0}, Real{-1}, Real{0}})};
}

/// Three triangles on the shared edge (0, 1), which no fan crosses.
template <typename Index, typename Real>
auto diagnostics_three_on_one_edge() -> tf::cpp::test::owned_mesh<Index, Real> {
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 1, 2, 0, 1, 3, 0, 1, 4},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}, Real{0}, Real{-1}, Real{0}, Real{0}, Real{0}, Real{1}})};
}

template <typename Index>
auto diagnostics_block(
    const tf::cpp::offset_blocked_buffer<Index, Index> &value, int index)
    -> std::vector<Index> {
  const auto block = value.get(index);
  return std::vector<Index>(block.begin(), block.end());
}

} // namespace

TEMPLATE_TEST_CASE("non_manifold_vertices names the vertices that are not one "
                   "fan",
                   "[cpp][topology][diagnostics]", float, double) {
  const auto bowtie = diagnostics_bowtie<tf::cpp::default_index_t, TestType>();
  const auto clean = tf::cpp::test::tetrahedron_mesh<TestType>();

  const auto vertices = tf::cpp::non_manifold_vertices(bowtie.mesh());

  static_assert(
      std::is_same_v<decltype(vertices),
                     const tf::cpp::nd_array<tf::cpp::default_index_t>>);
  REQUIRE((vertices.raw_shape() == tf::small_vector<int, 3>{1}));
  CHECK(vertices[0] == 0);
  CHECK(tf::cpp::is_non_manifold(bowtie.mesh()));

  CHECK(tf::cpp::non_manifold_vertices(clean.mesh()).empty());
  CHECK(tf::cpp::is_manifold(clean.mesh()));
}

// An edge three faces carry is crossed by no fan, so the vertices holding it
// stay named and stay as they were.
TEMPLATE_TEST_CASE("non_manifold_vertices names both ends of a shared edge",
                   "[cpp][topology][diagnostics]", float, double) {
  const auto owned =
      diagnostics_three_on_one_edge<tf::cpp::default_index_t, TestType>();

  const auto vertices = tf::cpp::non_manifold_vertices(owned.mesh());

  REQUIRE((vertices.raw_shape() == tf::small_vector<int, 3>{2}));
  CHECK(vertices[0] == 0);
  CHECK(vertices[1] == 1);

  const auto split = tf::cpp::split_non_manifold_vertices(owned.mesh());
  CHECK(split.mesh.points().size() == 5);
  CHECK(split.point_map.length() == 5);
}

TEMPLATE_TEST_CASE("split_non_manifold_vertices gives every fan a vertex",
                   "[cpp][topology][diagnostics][split]", float, double) {
  const auto owned = diagnostics_bowtie<tf::cpp::default_index_t, TestType>();

  const auto split = tf::cpp::split_non_manifold_vertices(owned.mesh());

  static_assert(
      std::is_same_v<
          decltype(split.mesh),
          tf::polygons_buffer<tf::cpp::default_index_t, TestType, 3, 3>>);

  // Face 0 is the smallest at the apex, so its fan keeps vertex 0.
  REQUIRE(split.mesh.faces().size() == 2);
  CHECK(split.mesh.faces()[0][0] == 0);
  CHECK(split.mesh.faces()[0][1] == 1);
  CHECK(split.mesh.faces()[0][2] == 2);
  CHECK(split.mesh.faces()[1][0] == 5);
  CHECK(split.mesh.faces()[1][1] == 3);
  CHECK(split.mesh.faces()[1][2] == 4);

  REQUIRE((split.point_map.raw_shape() == tf::small_vector<int, 3>{6}));
  CHECK(tf::cpp::test::has_values<tf::cpp::default_index_t>(
      split.point_map, {0, 1, 2, 3, 4, 0}));

  REQUIRE(split.mesh.points().size() == 6);
  for (std::size_t axis = 0; axis < 3; ++axis)
    CHECK(split.mesh.points()[5][axis] == split.mesh.points()[0][axis]);

  const auto repaired = tf::cpp::test::operand_of(split.mesh);
  CHECK(tf::cpp::is_manifold(repaired.mesh()));
  CHECK(tf::cpp::non_manifold_vertices(repaired.mesh()).empty());
}

TEMPLATE_TEST_CASE("split_non_manifold_vertices takes a mixed mesh at its own "
                   "arity",
                   "[cpp][topology][diagnostics][mixed]", float, double) {
  const auto owned =
      diagnostics_mixed_bowtie<tf::cpp::default_index_t, TestType>();

  const auto split = tf::cpp::split_non_manifold_vertices(owned.mesh());

  static_assert(std::is_same_v<decltype(split.mesh),
                               tf::polygons_buffer<tf::cpp::default_index_t,
                                                   TestType, 3,
                                                   tf::dynamic_size>>);
  REQUIRE(split.mesh.faces().size() == 2);
  CHECK(split.mesh.faces()[0].size() == 3);
  CHECK(split.mesh.faces()[0][0] == 0);
  CHECK(split.mesh.faces()[1][0] == 5);
  CHECK(split.point_map.length() == 6);
  CHECK(split.point_map[5] == 0);
}

TEMPLATE_TEST_CASE("boundary_rims walks each rim and says whether it closes",
                   "[cpp][topology][diagnostics][rims]", float, double) {
  const auto triangle = tf::cpp::test::triangle_mesh<TestType>();

  const auto rims = tf::cpp::boundary_rims(triangle.mesh());

  // The boundary is stated face by face and corner by corner, so the first
  // boundary edge is (face[2], face[0]) and the rim starts there.
  REQUIRE(rims.vertices.size() == 1);
  REQUIRE(rims.faces.size() == 1);
  REQUIRE((rims.closed.raw_shape() == tf::small_vector<int, 3>{1}));
  CHECK(rims.closed[0] != 0);
  CHECK((diagnostics_block(rims.vertices, 0) ==
         std::vector<tf::cpp::default_index_t>{2, 0, 1}));
  CHECK((diagnostics_block(rims.faces, 0) ==
         std::vector<tf::cpp::default_index_t>{0, 0, 0}));

  const auto closed = tf::cpp::test::tetrahedron_mesh<TestType>();
  const auto none = tf::cpp::boundary_rims(closed.mesh());
  CHECK(none.vertices.size() == 0);
  CHECK(none.faces.size() == 0);
  CHECK(none.closed.empty());
}

// A rim ends where the boundary stops passing straight through, so a rim of
// n vertices has n - 1 edges when it does not close.
TEMPLATE_TEST_CASE("boundary_rims states an open rim's edges",
                   "[cpp][topology][diagnostics][rims]", float, double) {
  const auto owned =
      diagnostics_three_on_one_edge<tf::cpp::default_index_t, TestType>();

  const auto rims = tf::cpp::boundary_rims(owned.mesh());

  REQUIRE(rims.vertices.size() == 3);
  REQUIRE(rims.faces.size() == 3);
  REQUIRE(rims.closed.length() == 3);
  for (int rim = 0; rim < 3; ++rim) {
    CHECK(rims.closed[static_cast<std::size_t>(rim)] == 0);
    CHECK(rims.vertices.get(rim).length() == 3);
    CHECK(rims.faces.get(rim).length() == 2);
  }
  CHECK((diagnostics_block(rims.vertices, 0) ==
         std::vector<tf::cpp::default_index_t>{1, 2, 0}));
  CHECK((diagnostics_block(rims.faces, 0) ==
         std::vector<tf::cpp::default_index_t>{0, 0}));
}

// AN ENTRY READS THE CACHE'S STRUCTURES: the three stand on one face
// membership and a second ask of any of them costs nothing.
TEMPLATE_TEST_CASE("the diagnostics family reuses one face membership",
                   "[cpp][topology][diagnostics][cache]", float, double) {
  const auto owned = diagnostics_bowtie<tf::cpp::default_index_t, TestType>();
  REQUIRE_FALSE(owned.cache.is_face_membership_built());

  static_cast<void>(tf::cpp::non_manifold_vertices(owned.mesh()));
  REQUIRE(owned.cache.face_membership_build_count() == 1);

  static_cast<void>(tf::cpp::boundary_rims(owned.mesh()));
  static_cast<void>(tf::cpp::split_non_manifold_vertices(owned.mesh()));
  static_cast<void>(tf::cpp::non_manifold_vertices(owned.mesh()));
  CHECK(owned.cache.face_membership_build_count() == 1);
  CHECK(owned.cache.is_face_membership_fresh(owned.mesh().geometry()));
}

TEMPLATE_TEST_CASE("async diagnostics answer what the synchronous entries do",
                   "[cpp][topology][diagnostics][async]", float, double) {
  const auto owned = diagnostics_bowtie<tf::cpp::default_index_t, TestType>();
  const auto expected_vertices = tf::cpp::non_manifold_vertices(owned.mesh());
  const auto expected_split =
      tf::cpp::split_non_manifold_vertices(owned.mesh());
  const auto expected_rims = tf::cpp::boundary_rims(owned.mesh());

  auto pending_vertices = tf::cpp::async::non_manifold_vertices(owned.mesh());
  auto pending_split =
      tf::cpp::async::split_non_manifold_vertices(owned.mesh());
  auto pending_rims = tf::cpp::async::boundary_rims(owned.mesh());
  static_assert(
      std::is_same_v<
          decltype(pending_vertices),
          std::future<tf::cpp::nd_array<tf::cpp::default_index_t>>>);
  static_assert(
      std::is_same_v<decltype(pending_rims),
                     std::future<tf::cpp::boundary_rims_result<
                         tf::cpp::default_index_t>>>);

  const auto vertices = pending_vertices.get();
  const auto split = pending_split.get();
  const auto rims = pending_rims.get();
  REQUIRE(vertices.raw_shape() == expected_vertices.raw_shape());
  CHECK(vertices[0] == expected_vertices[0]);
  REQUIRE(split.point_map.raw_shape() == expected_split.point_map.raw_shape());
  CHECK(split.mesh.faces()[1][0] == expected_split.mesh.faces()[1][0]);
  REQUIRE(rims.closed.raw_shape() == expected_rims.closed.raw_shape());
  CHECK(diagnostics_block(rims.vertices, 0) ==
        diagnostics_block(expected_rims.vertices, 0));
}
