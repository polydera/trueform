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

#include "trueform/cpp/core/build_winding_moments.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/spatial/async/signed_distance.hpp"
#include "trueform/cpp/spatial/signed_distance.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <future>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real>
auto point(Real x, Real y, Real z) -> tf::cpp::primitive<Real> {
  return tf::cpp::primitive<Real>(tf::cpp::primitive_kind::point,
                                  make_array<Real>({x, y, z}, {3}));
}

template <typename Real>
auto point_batch(std::initializer_list<Real> values, int count)
    -> tf::cpp::primitive<Real> {
  return tf::cpp::primitive<Real>(tf::cpp::primitive_kind::point,
                                  make_array<Real>(values, {count, 3}));
}

/// The unit cube spanning [-0.5, 0.5], wound outward.
template <typename Real>
auto cube_mesh() -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::make_box_mesh<tf::cpp::default_index_t, Real>(
      Real{1}, Real{1}, Real{1})};
}

template <typename Real>
auto empty_mesh() -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {};
}

template <typename Real>
auto translation(Real x, Real y, Real z) -> std::array<Real, 16> {
  return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
}

template <typename Real> constexpr auto tolerance() -> double {
  return std::is_same<Real, float>::value ? 1e-5 : 1e-11;
}

struct counting_resolver {
  std::shared_ptr<int> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    ++*submissions;
    return std::make_shared<state_type<T>>();
  }
};

template <typename Mesh, typename Query, typename = void>
struct signed_distance_takes : std::false_type {};

template <typename Mesh, typename Query>
struct signed_distance_takes<
    Mesh, Query,
    std::void_t<decltype(tf::cpp::signed_distance(
        std::declval<const Mesh &>(), std::declval<const Query &>()))>>
    : std::true_type {};

/// A winding number is of a three-dimensional surface, so the refusal is the
/// entry's own substitution — the query stands at the mesh's own dimension, so
/// nothing but the dimension itself can refuse.
static_assert(
    signed_distance_takes<tf::cpp::mesh<tf::cpp::default_index_t, float>,
                          tf::cpp::primitive<float, 3>>::value);
static_assert(
    !signed_distance_takes<tf::cpp::mesh<tf::cpp::default_index_t, float, 2>,
                           tf::cpp::primitive<float, 2>>::value);

} // namespace

TEMPLATE_TEST_CASE("signed distances are negative inside and positive outside",
                   "[cpp][spatial][signed-distance]", float, double) {
  const auto cube_storage = cube_mesh<TestType>();
  const auto cube = cube_storage.mesh();

  CHECK(tf::cpp::signed_distance(cube, point<TestType>(0, 0, 0)).scalar() ==
        Catch::Approx(-0.5).margin(tolerance<TestType>()));
  CHECK(tf::cpp::signed_distance(cube, point<TestType>(2, 0, 0)).scalar() ==
        Catch::Approx(1.5).margin(tolerance<TestType>()));

  const auto batch = tf::cpp::signed_distance(
      cube, point_batch<TestType>({0, 0, 0, 2, 0, 0, 0, 0, 3}, 3));
  REQUIRE(batch.is_batch());
  const auto values = batch.batch();
  REQUIRE(values.length() == 3);
  CHECK(values[0] == Catch::Approx(-0.5).margin(tolerance<TestType>()));
  CHECK(values[1] == Catch::Approx(1.5).margin(tolerance<TestType>()));
  CHECK(values[2] == Catch::Approx(2.5).margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE("signed distances honor transformations",
                   "[cpp][spatial][signed-distance][transform]", float,
                   double) {
  auto cube_storage = cube_mesh<TestType>();
  cube_storage.place(translation<TestType>(10, 0, 0));
  const auto cube = cube_storage.mesh();

  CHECK(tf::cpp::signed_distance(cube, point<TestType>(10, 0, 0)).scalar() ==
        Catch::Approx(-0.5).margin(tolerance<TestType>()));
  CHECK(tf::cpp::signed_distance(cube, point<TestType>(12, 0, 0)).scalar() ==
        Catch::Approx(1.5).margin(tolerance<TestType>()));
}

TEST_CASE("signed distances reuse fresh winding moments and rebuild stale ones",
          "[cpp][spatial][signed-distance][winding-moments]") {
  auto cube_storage = cube_mesh<float>();
  const auto query = point<float>(0, 0, 0);
  const auto cube = cube_storage.mesh();
  CHECK_FALSE(cube_storage.cache.is_winding_moments_built());

  CHECK(tf::cpp::signed_distance(cube, query).scalar() ==
        Catch::Approx(-0.5).margin(1e-5));
  CHECK(cube_storage.cache.is_winding_moments_fresh(cube.geometry()));
  CHECK(cube_storage.cache.winding_moments_build_count() == 1);
  CHECK(cube_storage.cache.tree_build_count() == 1);
  static_cast<void>(tf::cpp::signed_distance(cube, query));
  CHECK(cube_storage.cache.winding_moments_build_count() == 1);

  // the caller states the change, and assembles again for the reading it made
  for (auto point : cube_storage.polygons.points())
    for (auto &coordinate : point)
      coordinate *= 2;
  cube_storage.cache.points_changed();
  const auto scaled = cube_storage.mesh();
  CHECK_FALSE(cube_storage.cache.is_winding_moments_fresh(scaled.geometry()));
  CHECK(tf::cpp::signed_distance(scaled, query).scalar() ==
        Catch::Approx(-1).margin(1e-5));
  CHECK(cube_storage.cache.tree_build_count() == 2);
  CHECK(cube_storage.cache.winding_moments_build_count() == 2);
}

TEST_CASE("a prebuilt mesh answers signed distances without building again",
          "[cpp][spatial][signed-distance][winding-moments]") {
  const auto cube_storage = cube_mesh<float>();
  const auto cube = cube_storage.mesh();
  tf::cpp::build_winding_moments(cube);
  REQUIRE(cube_storage.cache.tree_build_count() == 1);
  REQUIRE(cube_storage.cache.winding_moments_build_count() == 1);

  CHECK(tf::cpp::signed_distance(cube, point<float>(0, 0, 2)).scalar() ==
        Catch::Approx(1.5).margin(1e-5));
  CHECK(cube_storage.cache.winding_moments_build_count() == 1);
}

TEST_CASE("signed distances refuse a non-point query and an empty form",
          "[cpp][spatial][signed-distance][validation]") {
  const auto cube_storage = cube_mesh<float>();
  const auto cube = cube_storage.mesh();
  const auto segment =
      tf::cpp::primitive<float>(tf::cpp::primitive_kind::segment,
                                make_array<float>({0, 0, 0, 1, 0, 0}, {2, 3}));
  CHECK_THROWS_AS(tf::cpp::signed_distance(cube, segment),
                  std::invalid_argument);

  const auto empty_storage = empty_mesh<float>();
  CHECK_THROWS_AS(
      tf::cpp::signed_distance(empty_storage.mesh(), point<float>(0, 0, 0)),
      std::invalid_argument);

  // an empty query batch answers itself, so the form is never read for it
  tf::buffer<float> no_coordinates;
  no_coordinates.allocate(0);
  const auto empty_batch = tf::cpp::primitive<float>(
      tf::cpp::primitive_kind::point,
      tf::cpp::nd_array<float>::from_buffer(std::move(no_coordinates), {0, 3}));
  const auto unread_storage = cube_mesh<float>();
  const auto answered =
      tf::cpp::signed_distance(unread_storage.mesh(), empty_batch);
  REQUIRE(answered.is_batch());
  CHECK(answered.batch().empty());
  CHECK_FALSE(unread_storage.cache.is_winding_moments_built());
}

TEST_CASE("async signed distances preserve scalar batch and error behavior",
          "[cpp][spatial][signed-distance][async]") {
  const auto cube_storage = cube_mesh<float>();
  const auto cube = cube_storage.mesh();
  auto query = point<double>(0, 0, 0);
  auto batch = point_batch<double>({0, 0, 0, 2, 0, 0}, 2);

  // a cache shared by concurrent jobs is filled before it is shared
  tf::cpp::build_winding_moments(cube);
  const auto builds = cube_storage.cache.winding_moments_build_count();

  auto scalar = tf::cpp::async::signed_distance(cube, query);
  auto batched = tf::cpp::async::signed_distance(cube, batch);
  STATIC_REQUIRE(std::is_same_v<decltype(scalar),
                                std::future<tf::cpp::distance_result<float>>>);
  CHECK(scalar.get().scalar() == Catch::Approx(-0.5).margin(1e-5));
  const auto values = batched.get().batch();
  REQUIRE(values.length() == 2);
  CHECK(values[0] == Catch::Approx(-0.5).margin(1e-5));
  CHECK(values[1] == Catch::Approx(1.5).margin(1e-5));
  CHECK(cube_storage.cache.winding_moments_build_count() == builds);

  const auto submissions = std::make_shared<int>(0);
  auto custom = tf::cpp::async::signed_distance(counting_resolver{submissions},
                                                cube, query);
  CHECK(*submissions == 1);
  CHECK(custom.get().scalar() == Catch::Approx(-0.5).margin(1e-5));

  const auto vector_query = tf::cpp::primitive<float>(
      tf::cpp::primitive_kind::vector, make_array<float>({1, 0, 0}, {3}));
  auto failure = tf::cpp::async::signed_distance(cube, vector_query);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}
