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

#include "trueform/cpp/geometry/async/dihedral_angles.hpp"
#include "trueform/cpp/geometry/async/face_quality.hpp"
#include "trueform/cpp/geometry/dihedral_angles.hpp"
#include "trueform/cpp/geometry/face_quality.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <type_traits>
#include <utility>

namespace {

constexpr long double quality_pi = 3.141592653589793238462643383279502884L;

template <typename Real> auto quality_tolerance() -> Real {
  return std::is_same_v<Real, float> ? Real{1e-4} : Real{1e-12};
}

/// An equilateral triangle beside a right isoceles one, in that face order.
template <typename Index, typename Real>
auto make_quality_triangles() -> tf::cpp::test::owned_mesh<Index, Real> {
  const auto height = std::sqrt(Real{3}) / Real{2};
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 1, 2, 3, 4, 5},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0.5}, height,
       Real{0}, Real{0}, Real{0}, Real{4}, Real{1}, Real{0}, Real{4}, Real{0},
       Real{1}, Real{4}})};
}

/// One square face, so the triangle measure has nothing to say about it.
template <typename Index, typename Real>
auto make_quality_quad()
    -> tf::cpp::test::owned_mesh<Index, Real, 3, tf::dynamic_size> {
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 4}, {0, 1, 2, 3},
      {Real{0}, Real{0}, Real{0}, Real{2}, Real{0}, Real{0}, Real{2}, Real{1},
       Real{0}, Real{0}, Real{1}, Real{0}})};
}

template <typename Real>
auto quality_degrees(Real radians) -> Real {
  return radians * Real{180} / static_cast<Real>(quality_pi);
}

} // namespace

TEMPLATE_TEST_CASE("face_quality measures each face by hand-derived numbers",
                   "[cpp][geometry][quality]", float, double) {
  const auto owned =
      make_quality_triangles<tf::cpp::default_index_t, TestType>();

  const auto measured = tf::cpp::face_quality(owned.mesh());

  static_assert(std::is_same_v<decltype(measured.quality),
                               tf::cpp::nd_array<TestType>>);
  REQUIRE((measured.quality.raw_shape() == tf::small_vector<int, 3>{2}));
  REQUIRE(measured.min_angle.raw_shape() == measured.quality.raw_shape());
  REQUIRE(measured.max_angle.raw_shape() == measured.quality.raw_shape());
  REQUIRE(measured.aspect_ratio.raw_shape() == measured.quality.raw_shape());

  const auto tolerance = quality_tolerance<TestType>();

  CHECK(std::abs(measured.quality[0] - TestType{1}) < tolerance);
  CHECK(std::abs(quality_degrees(measured.min_angle[0]) - TestType{60}) <
        TestType{1e-2});
  CHECK(std::abs(quality_degrees(measured.max_angle[0]) - TestType{60}) <
        TestType{1e-2});
  CHECK(std::abs(measured.aspect_ratio[0] - TestType{1}) < tolerance);

  // 2A = 1 over a longest side of 2, so the measure is (2/sqrt3) / 2
  CHECK(std::abs(measured.quality[1] - TestType{0.5773502691896258}) <
        tolerance);
  CHECK(std::abs(quality_degrees(measured.min_angle[1]) - TestType{45}) <
        TestType{1e-2});
  CHECK(std::abs(quality_degrees(measured.max_angle[1]) - TestType{90}) <
        TestType{1e-2});
  CHECK(std::abs(measured.aspect_ratio[1] - std::sqrt(TestType{2})) <
        tolerance);
}

// The angles and the aspect ratio hold at any arity; the triangle measure is a
// triangle's alone, so a quad reads -1 for it.
TEMPLATE_TEST_CASE("face_quality takes a mixed mesh at its own arity",
                   "[cpp][geometry][quality][mixed]", float, double) {
  const auto owned = make_quality_quad<tf::cpp::default_index_t, TestType>();

  const auto measured = tf::cpp::face_quality(owned.mesh());

  REQUIRE((measured.quality.raw_shape() == tf::small_vector<int, 3>{1}));
  CHECK(measured.quality[0] == TestType{-1});
  CHECK(std::abs(quality_degrees(measured.min_angle[0]) - TestType{90}) <
        TestType{1e-2});
  CHECK(std::abs(quality_degrees(measured.max_angle[0]) - TestType{90}) <
        TestType{1e-2});
  CHECK(std::abs(measured.aspect_ratio[0] - TestType{2}) <
        quality_tolerance<TestType>());
}

TEMPLATE_TEST_CASE("face_quality answers the empty mesh",
                   "[cpp][geometry][quality][empty]", float, double) {
  const tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType> empty;

  const auto measured = tf::cpp::face_quality(empty.mesh());

  CHECK(measured.quality.empty());
  CHECK(measured.min_angle.empty());
  CHECK(measured.max_angle.empty());
  CHECK(measured.aspect_ratio.empty());
}

// A box turns through a right angle across its twelve edges and through
// nothing across the six diagonals splitting its square faces.
TEMPLATE_TEST_CASE("dihedral_angles measures every edge two faces share",
                   "[cpp][geometry][quality][dihedral]", float, double) {
  const auto owned = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2}));

  const auto measured = tf::cpp::dihedral_angles(owned.mesh());

  static_assert(std::is_same_v<decltype(measured.edges),
                               tf::cpp::nd_array<std::int32_t>>);
  REQUIRE((measured.edges.raw_shape() == tf::small_vector<int, 3>{18, 2}));
  REQUIRE((measured.angles.raw_shape() == tf::small_vector<int, 3>{18}));

  int flat = 0;
  int right = 0;
  for (std::size_t index = 0; index < measured.angles.length(); ++index) {
    const auto degrees = quality_degrees(measured.angles[index]);
    if (std::abs(degrees) < TestType{1e-2})
      ++flat;
    else if (std::abs(degrees - TestType{90}) < TestType{1e-2})
      ++right;
    CHECK(measured.edges[2 * index] < measured.edges[2 * index + 1]);
  }
  CHECK(flat == 6);
  CHECK(right == 12);
}

// A boundary edge joins no pair of faces and turns through no angle, so two
// triangles sharing one edge state that edge and nothing else.
TEMPLATE_TEST_CASE("dihedral_angles states the shared edge alone",
                   "[cpp][geometry][quality][dihedral]", float, double) {
  const tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType> owned{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2, 1, 0, 3},
          {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
           TestType{0}, TestType{0}, TestType{1}, TestType{0}, TestType{0},
           TestType{0}, TestType{1}})};

  const auto measured = tf::cpp::dihedral_angles(owned.mesh());

  REQUIRE((measured.edges.raw_shape() == tf::small_vector<int, 3>{1, 2}));
  CHECK(measured.edges[0] == 0);
  CHECK(measured.edges[1] == 1);
  CHECK(std::abs(quality_degrees(measured.angles[0]) - TestType{90}) <
        TestType{1e-2});
}

// AN ENTRY READS THE CACHE'S STRUCTURES: the edge link and the face normals
// are built once and the second ask costs nothing.
TEMPLATE_TEST_CASE("dihedral_angles reuses the edge link and the normals",
                   "[cpp][geometry][quality][cache]", float, double) {
  const auto owned = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2}));
  REQUIRE_FALSE(owned.cache.is_manifold_edge_link_built());
  REQUIRE_FALSE(owned.cache.is_face_normals_built());

  static_cast<void>(tf::cpp::dihedral_angles(owned.mesh()));
  REQUIRE(owned.cache.manifold_edge_link_build_count() == 1);
  REQUIRE(owned.cache.face_normals_build_count() == 1);

  static_cast<void>(tf::cpp::dihedral_angles(owned.mesh()));
  CHECK(owned.cache.manifold_edge_link_build_count() == 1);
  CHECK(owned.cache.face_normals_build_count() == 1);
  CHECK(owned.cache.is_manifold_edge_link_fresh(owned.mesh().geometry()));
  CHECK(owned.cache.is_face_normals_fresh(owned.mesh().geometry()));
}

TEMPLATE_TEST_CASE("async quality answers what the synchronous entries do",
                   "[cpp][geometry][quality][async]", float, double) {
  const auto triangles =
      make_quality_triangles<tf::cpp::default_index_t, TestType>();
  const auto box = tf::cpp::test::operand_of(
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2}));
  const auto expected_quality = tf::cpp::face_quality(triangles.mesh());
  const auto expected_angles = tf::cpp::dihedral_angles(box.mesh());

  auto pending_quality = tf::cpp::async::face_quality(triangles.mesh());
  auto pending_angles = tf::cpp::async::dihedral_angles(box.mesh());
  static_assert(
      std::is_same_v<decltype(pending_quality),
                     std::future<tf::cpp::face_quality_result<TestType>>>);
  using angles_result = tf::cpp::dihedral_angles_result<std::int32_t, TestType>;
  static_assert(std::is_same_v<decltype(pending_angles),
                               std::future<angles_result>>);

  const auto quality = pending_quality.get();
  const auto angles = pending_angles.get();
  REQUIRE(quality.quality.raw_shape() == expected_quality.quality.raw_shape());
  for (std::size_t index = 0; index < quality.quality.length(); ++index)
    CHECK(quality.quality[index] == expected_quality.quality[index]);
  REQUIRE(angles.angles.raw_shape() == expected_angles.angles.raw_shape());
  for (std::size_t index = 0; index < angles.angles.length(); ++index)
    CHECK(angles.angles[index] == expected_angles.angles[index]);
}
