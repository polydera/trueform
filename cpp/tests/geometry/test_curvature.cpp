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
#include "carriers.hpp"

#include "trueform/cpp/core/build_face_membership.hpp"
#include "trueform/cpp/core/build_point_normals.hpp"
#include "trueform/cpp/core/build_vertex_link.hpp"
#include "trueform/cpp/geometry/async/principal_curvatures.hpp"
#include "trueform/cpp/geometry/async/principal_directions.hpp"
#include "trueform/cpp/geometry/async/shape_index.hpp"
#include "trueform/cpp/geometry/make_plane_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/principal_curvatures.hpp"
#include "trueform/cpp/geometry/principal_directions.hpp"
#include "trueform/cpp/geometry/shape_index.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace {

using curvature_index = tf::cpp::default_index_t;

template <typename Real>
using curvature_owned = tf::cpp::test::owned_mesh<curvature_index, Real>;

template <typename Real> auto sphere_mesh() -> curvature_owned<Real> {
  return {tf::cpp::make_sphere_mesh<curvature_index>(Real{2}, std::int32_t{12},
                                                     std::int32_t{12})};
}

template <typename Real> auto plane_mesh() -> curvature_owned<Real> {
  return {tf::cpp::make_plane_mesh<curvature_index>(
      Real{4}, Real{4}, std::int32_t{4}, std::int32_t{4})};
}

template <typename Real>
auto direction_norm(const tf::cpp::nd_array<Real> &directions, int index)
    -> double {
  auto squared_norm = 0.0;
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    const auto value = static_cast<double>(
        directions[static_cast<std::size_t>(index * 3 + coordinate)]);
    squared_norm += value * value;
  }
  return std::sqrt(squared_norm);
}

template <typename Real>
auto direction_dot(const tf::cpp::nd_array<Real> &first,
                   const tf::cpp::nd_array<Real> &second, int index) -> double {
  auto dot = 0.0;
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    const auto offset = static_cast<std::size_t>(index * 3 + coordinate);
    dot += static_cast<double>(first[offset]) *
           static_cast<double>(second[offset]);
  }
  return dot;
}

template <typename Real>
auto matches(const tf::cpp::principal_curvatures_result<Real> &actual,
             const tf::cpp::principal_curvatures_result<Real> &expected,
             double tolerance) -> bool {
  if (actual.k0.raw_shape() != expected.k0.raw_shape() ||
      actual.k1.raw_shape() != expected.k1.raw_shape())
    return false;
  for (std::size_t index = 0; index < actual.k0.length(); ++index) {
    const auto expected_k0 = static_cast<double>(expected.k0[index]);
    const auto expected_k1 = static_cast<double>(expected.k1[index]);
    const auto k0_scale = std::max(1.0, std::abs(expected_k0));
    const auto k1_scale = std::max(1.0, std::abs(expected_k1));
    if (std::abs(static_cast<double>(actual.k0[index]) - expected_k0) >
            tolerance * k0_scale ||
        std::abs(static_cast<double>(actual.k1[index]) - expected_k1) >
            tolerance * k1_scale)
      return false;
  }
  return true;
}

template <typename Real>
auto matches(const tf::cpp::nd_array<Real> &actual,
             const tf::cpp::nd_array<Real> &expected, double tolerance)
    -> bool {
  if (actual.raw_shape() != expected.raw_shape())
    return false;
  for (std::size_t index = 0; index < actual.length(); ++index) {
    const auto expected_value = static_cast<double>(expected[index]);
    const auto scale = std::max(1.0, std::abs(expected_value));
    if (std::abs(static_cast<double>(actual[index]) - expected_value) >
        tolerance * scale)
      return false;
  }
  return true;
}

struct curvature_counting_resolver {
  std::shared_ptr<std::atomic<int>> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<state_type<T>>();
  }
};

} // namespace

TEMPLATE_TEST_CASE(
    "curvature APIs return finite typed arrays with canonical shapes",
    "[cpp][geometry][curvature]", float, double) {
  const auto owned = sphere_mesh<TestType>();
  const auto number_of_points =
      static_cast<int>(owned.mesh().number_of_points());

  const auto curvatures = tf::cpp::principal_curvatures(owned.mesh());
  const auto directions = tf::cpp::principal_directions(owned.mesh());
  const auto indices = tf::cpp::shape_index(owned.mesh());

  static_assert(
      std::is_same_v<decltype(curvatures.k0), tf::cpp::nd_array<TestType>>);
  static_assert(
      std::is_same_v<decltype(curvatures.k1), tf::cpp::nd_array<TestType>>);
  static_assert(
      std::is_same_v<decltype(directions.d0), tf::cpp::nd_array<TestType>>);
  static_assert(
      std::is_same_v<decltype(directions.d1), tf::cpp::nd_array<TestType>>);
  static_assert(
      std::is_same_v<decltype(indices), const tf::cpp::nd_array<TestType>>);

  CHECK((curvatures.k0.raw_shape() ==
         tf::small_vector<int, 3>{number_of_points}));
  CHECK((curvatures.k1.raw_shape() ==
         tf::small_vector<int, 3>{number_of_points}));
  CHECK((directions.k0.raw_shape() ==
         tf::small_vector<int, 3>{number_of_points}));
  CHECK((directions.k1.raw_shape() ==
         tf::small_vector<int, 3>{number_of_points}));
  CHECK((directions.d0.raw_shape() ==
         tf::small_vector<int, 3>{number_of_points, 3}));
  CHECK((directions.d1.raw_shape() ==
         tf::small_vector<int, 3>{number_of_points, 3}));
  CHECK((indices.raw_shape() == tf::small_vector<int, 3>{number_of_points}));

  auto mean_k0 = 0.0;
  auto mean_k1 = 0.0;
  auto mean_shape_index = 0.0;
  const auto direction_tolerance =
      std::is_same_v<TestType, float> ? 1e-4 : 1e-10;
  for (int index = 0; index < number_of_points; ++index) {
    const auto k0 = static_cast<double>(curvatures.k0[index]);
    const auto k1 = static_cast<double>(curvatures.k1[index]);
    const auto shape = static_cast<double>(indices[index]);
    CHECK(std::isfinite(k0));
    CHECK(std::isfinite(k1));
    CHECK(std::isfinite(shape));
    CHECK(k0 >= k1);
    CHECK(shape >= -1.0);
    CHECK(shape <= 1.0);
    CHECK(direction_norm(directions.d0, index) ==
          Catch::Approx(1.0).margin(direction_tolerance));
    CHECK(direction_norm(directions.d1, index) ==
          Catch::Approx(1.0).margin(direction_tolerance));
    CHECK(direction_dot(directions.d0, directions.d1, index) ==
          Catch::Approx(0.0).margin(direction_tolerance));
    mean_k0 += k0;
    mean_k1 += k1;
    mean_shape_index += shape;
  }
  mean_k0 /= number_of_points;
  mean_k1 /= number_of_points;
  mean_shape_index /= number_of_points;

  CHECK(mean_k0 == Catch::Approx(0.5).margin(0.15));
  CHECK(mean_k1 == Catch::Approx(0.5).margin(0.15));
  CHECK(mean_shape_index > 0.5);
}

TEMPLATE_TEST_CASE("curvature preserves zero-ring and planar behavior",
                   "[cpp][geometry][curvature]", float, double) {
  const auto owned = plane_mesh<TestType>();

  const auto zero_ring = tf::cpp::principal_curvatures(owned.mesh(), 0);
  const auto planar = tf::cpp::principal_curvatures(owned.mesh());
  const auto indices = tf::cpp::shape_index(owned.mesh());

  const auto number_of_points =
      static_cast<int>(owned.mesh().number_of_points());
  for (int index = 0; index < number_of_points; ++index) {
    CHECK(zero_ring.k0[index] == TestType{0});
    CHECK(zero_ring.k1[index] == TestType{0});
    CHECK(planar.k0[index] == Catch::Approx(0.0).margin(1e-6));
    CHECK(planar.k1[index] == Catch::Approx(0.0).margin(1e-6));
    CHECK(std::isfinite(static_cast<double>(indices[index])));
    CHECK(indices[index] >= TestType{-1});
    CHECK(indices[index] <= TestType{1});
  }
}

TEMPLATE_TEST_CASE("curvature rejects negative rings before cache preparation",
                   "[cpp][geometry][curvature][validation]", float, double) {
  const auto owned = sphere_mesh<TestType>();

  CHECK_THROWS_AS(tf::cpp::principal_curvatures(owned.mesh(), -1),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::principal_directions(owned.mesh(), -1),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::shape_index(owned.mesh(), -1),
                  std::invalid_argument);
  CHECK_FALSE(owned.cache.is_face_membership_built());
  CHECK_FALSE(owned.cache.is_vertex_link_built());
}

// A default-assembled carrier is the EMPTY mesh: curvature answers it with
// empty arrays rather than refusing it.
TEMPLATE_TEST_CASE("curvature answers the empty mesh",
                   "[cpp][geometry][curvature][empty]", float, double) {
  const curvature_owned<TestType> empty;

  CHECK(tf::cpp::principal_curvatures(empty.mesh()).k0.empty());
  CHECK(tf::cpp::principal_directions(empty.mesh()).d0.empty());
  CHECK(tf::cpp::shape_index(empty.mesh()).empty());
}

TEMPLATE_TEST_CASE("curvature reuses the caches it filled",
                   "[cpp][geometry][curvature][cache]", float, double) {
  auto owned = sphere_mesh<TestType>();
  CHECK_FALSE(owned.cache.is_face_membership_built());
  CHECK_FALSE(owned.cache.is_vertex_link_built());

  static_cast<void>(tf::cpp::principal_curvatures(owned.mesh()));
  REQUIRE(owned.cache.is_face_membership_fresh(owned.mesh().geometry()));
  REQUIRE(owned.cache.is_vertex_link_fresh(owned.mesh().geometry()));
  CHECK(owned.cache.face_membership_build_count() == 1);
  const auto vertex_link_builds = owned.cache.vertex_link_build_count();

  static_cast<void>(tf::cpp::principal_directions(owned.mesh()));
  static_cast<void>(tf::cpp::shape_index(owned.mesh()));
  CHECK(owned.cache.vertex_link_build_count() == vertex_link_builds);
  CHECK(owned.cache.face_membership_build_count() == 1);

  // a point that moved stales what stands on where the points ARE and leaves
  // what stands on which points there are exactly where it was
  owned.polygons.points_buffer().data_buffer()[0] +=
      std::numeric_limits<TestType>::epsilon();
  owned.cache.points_changed();
  CHECK(owned.cache.is_face_membership_fresh(owned.mesh().geometry()));
  CHECK(owned.cache.is_vertex_link_fresh(owned.mesh().geometry()));

  static_cast<void>(tf::cpp::shape_index(owned.mesh()));
  CHECK(owned.cache.vertex_link_build_count() == vertex_link_builds);
  CHECK(owned.cache.face_membership_build_count() == 1);
}

TEMPLATE_TEST_CASE("curvature result arrays retain independent ownership",
                   "[cpp][geometry][curvature][ownership]", float, double) {
  auto owned = sphere_mesh<TestType>();
  auto result = tf::cpp::principal_directions(owned.mesh(), 1);
  const auto first_k1 = result.k1[0];
  const auto first_d1 = result.d1[0];

  CHECK(result.k0.raw_owner() != result.k1.raw_owner());
  CHECK(result.d0.raw_owner() != result.d1.raw_owner());
  owned = {};

  REQUIRE(result.k0.is_valid());
  REQUIRE(result.k1.is_valid());
  REQUIRE(result.d0.is_valid());
  REQUIRE(result.d1.is_valid());
  result.k0[0] += TestType{1};
  result.d0[0] += TestType{1};
  CHECK(result.k1[0] == first_k1);
  CHECK(result.d1[0] == first_d1);
}

TEMPLATE_TEST_CASE("async curvature answers through the cache it was handed",
                   "[cpp][geometry][curvature][async]", float, double) {
  auto owned = plane_mesh<TestType>();
  const auto expected_owned = plane_mesh<TestType>();
  const auto expected_curvatures =
      tf::cpp::principal_curvatures(expected_owned.mesh());
  const auto expected_directions =
      tf::cpp::principal_directions(expected_owned.mesh());
  const auto expected_indices = tf::cpp::shape_index(expected_owned.mesh());

  // THE CONTRACT: a cache three jobs will share is filled before it is shared,
  // and the verbs are that ask — the point normals, the membership they are
  // gathered through, and the vertex link the fit walks
  tf::cpp::build_point_normals(owned.mesh());
  tf::cpp::build_face_membership(owned.mesh());
  tf::cpp::build_vertex_link(owned.mesh());

  auto curvatures = tf::cpp::async::principal_curvatures(owned.mesh());
  auto directions = tf::cpp::async::principal_directions(owned.mesh());
  auto indices = tf::cpp::async::shape_index(owned.mesh());
  static_assert(std::is_same_v<
                decltype(curvatures),
                std::future<tf::cpp::principal_curvatures_result<TestType>>>);
  static_assert(std::is_same_v<
                decltype(directions),
                std::future<tf::cpp::principal_directions_result<TestType>>>);
  static_assert(std::is_same_v<decltype(indices),
                               std::future<tf::cpp::nd_array<TestType>>>);

  const auto actual_curvatures = curvatures.get();
  const auto actual_directions = directions.get();
  const auto actual_indices = indices.get();
  const auto tolerance = std::is_same_v<TestType, float> ? 1e-5 : 1e-12;
  CHECK(matches(actual_curvatures, expected_curvatures, tolerance));
  CHECK(matches(actual_directions.k0, expected_directions.k0, tolerance));
  CHECK(matches(actual_directions.k1, expected_directions.k1, tolerance));
  CHECK(matches(actual_directions.d0, expected_directions.d0, tolerance));
  CHECK(matches(actual_directions.d1, expected_directions.d1, tolerance));
  CHECK(matches(actual_indices, expected_indices, tolerance));
  // the jobs read the caller's own cache, so what the workers filled is there
  CHECK(owned.cache.is_face_membership_built());
  CHECK(owned.cache.is_vertex_link_built());

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::principal_curvatures(
      curvature_counting_resolver{submissions}, owned.mesh(), 0);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  const auto zero_ring = custom.get();
  CHECK(zero_ring.k0[0] == TestType{0});
  CHECK(zero_ring.k1[0] == TestType{0});

  auto failure = tf::cpp::async::shape_index(owned.mesh(), -1);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}
