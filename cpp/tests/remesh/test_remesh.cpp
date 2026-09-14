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

#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/remesh/async/decimated.hpp"
#include "trueform/cpp/remesh/async/isotropic_remeshed.hpp"
#include "trueform/cpp/remesh/async/simplified.hpp"
#include "trueform/cpp/remesh/decimated.hpp"
#include "trueform/cpp/remesh/isotropic_remeshed.hpp"
#include "trueform/cpp/remesh/simplified.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

struct counting_resolver {
  std::shared_ptr<std::atomic<int>> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<state_type<T>>();
  }
};

template <typename Real>
using owned_t = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename T>
auto filled_array(std::size_t count, T value) -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(count);
  for (auto &item : buffer)
    item = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer),
                                           {static_cast<int>(count)});
}

template <typename Real> auto translation(Real x) -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x,       Real{0}, Real{1},
          Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
          Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Real> auto scaling(Real x) -> std::array<Real, 16> {
  return {x,       Real{0}, Real{0}, Real{0}, Real{0}, Real{1},
          Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
          Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Real> auto sphere(int stacks) -> owned_t<Real> {
  owned_t<Real> owned;
  owned.polygons = tf::cpp::make_sphere_mesh(Real{1}, stacks, stacks);
  return owned;
}

template <typename Real> auto box(Real side) -> owned_t<Real> {
  owned_t<Real> owned;
  owned.polygons = tf::cpp::make_box_mesh(side, side, side);
  return owned;
}

template <typename Real> auto subdivided_box(Real side) -> owned_t<Real> {
  owned_t<Real> owned;
  owned.polygons = tf::cpp::make_box_mesh(side, side, side, 2, 2, 2);
  return owned;
}

/// One point and no faces: what every operation hands straight back.
template <typename Real> auto single_point() -> owned_t<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {}, {Real{2}, Real{3}, Real{4}})};
}

template <typename Real> auto first_coordinate(const owned_t<Real> &owned) {
  return owned.polygons.points_buffer().data_buffer()[0];
}

} // namespace

TEMPLATE_TEST_CASE("remesh owns each operation result and its regions",
                   "[cpp][remesh][sync]", float, double) {
  auto source = sphere<TestType>(8);
  auto regions = filled_array<std::int32_t>(source.polygons.size(), 7);

  tf::decimate_config<TestType> decimate;
  decimate.parallel = false;
  const auto decimated =
      tf::cpp::decimated(source.mesh(), TestType{0.7}, decimate, regions);

  tf::isotropic_remesh_config<TestType> isotropic(TestType{0.5}, 1);
  isotropic.parallel = false;
  const auto remeshed =
      tf::cpp::isotropic_remeshed(source.mesh(), isotropic, regions);

  tf::simplify_config<TestType> simplify(TestType{0.01});
  simplify.parallel = false;
  simplify.optimize_iterations = 1;
  const auto simplified = tf::cpp::simplified(source.mesh(), simplify, regions);

  static_assert(
      std::is_same_v<
          decltype(decimated),
          const tf::cpp::remesh_result<tf::cpp::default_index_t, TestType>>);
  CHECK(decimated.mesh.size() > 0);
  CHECK(decimated.regions.length() == decimated.mesh.size());
  CHECK(remeshed.mesh.size() > 0);
  CHECK(remeshed.regions.length() == remeshed.mesh.size());
  CHECK(simplified.mesh.size() > 0);
  CHECK(simplified.regions.length() == simplified.mesh.size());

  source = {};
  regions.destroy();
  CHECK(decimated.regions[0] == 7);
}

TEMPLATE_TEST_CASE("plain remesh results carry valid empty regions",
                   "[cpp][remesh][plain]", float, double) {
  auto source = box(TestType{2});

  const auto decimated = tf::cpp::decimated(source.mesh(), TestType{1});
  tf::isotropic_remesh_config<TestType> isotropic(TestType{1}, 0);
  const auto remeshed = tf::cpp::isotropic_remeshed(source.mesh(), isotropic);
  tf::simplify_config<TestType> simplify;
  simplify.iterations = 0;
  const auto simplified = tf::cpp::simplified(source.mesh(), simplify);

  CHECK(decimated.regions.is_valid());
  CHECK(decimated.regions.raw_shape() == tf::small_vector<int, 3>{0});
  CHECK(remeshed.regions.is_valid());
  CHECK(remeshed.regions.empty());
  CHECK(simplified.regions.is_valid());
  CHECK(simplified.regions.empty());
}

/// A placement is the operation's coordinate frame, and nothing more: the
/// result is core's own storage, in the coordinates the operand was authored
/// in, and the operand's own points never move.
TEMPLATE_TEST_CASE("remesh reads a placed mesh in its own coordinates",
                   "[cpp][remesh][transform]", float, double) {
  auto source = box(TestType{2});
  source.place(translation(TestType{10}));

  const auto result = tf::cpp::decimated(source.mesh(), TestType{1});

  CHECK(result.mesh.points_buffer().data_buffer()[0] ==
        Catch::Approx(TestType{-1}));
  CHECK(first_coordinate(source) == Catch::Approx(TestType{-1}));
}

TEMPLATE_TEST_CASE("remesh defines placed empty mesh behavior",
                   "[cpp][remesh][empty]", float, double) {
  auto source = single_point<TestType>();
  source.place(translation(TestType{5}));
  auto regions = filled_array<std::int32_t>(0, 3);

  const auto decimated =
      tf::cpp::decimated(source.mesh(), TestType{0.5}, {}, regions);
  const auto remeshed = tf::cpp::isotropic_remeshed(
      source.mesh(), tf::isotropic_remesh_config<TestType>(TestType{1}),
      regions);
  const auto simplified = tf::cpp::simplified(source.mesh(), {}, regions);

  for (const auto *result : {&decimated, &remeshed, &simplified}) {
    CHECK(result->mesh.size() == 0);
    CHECK(result->mesh.points_buffer().size() == 1);
    CHECK(result->mesh.points_buffer().data_buffer()[0] ==
          Catch::Approx(TestType{2}));
    CHECK(result->regions.is_valid());
    CHECK(result->regions.empty());
  }
}

TEMPLATE_TEST_CASE("remesh validates meshes arrays and options",
                   "[cpp][remesh][validation]", float, double) {
  // a mesh over empty storage is the EMPTY mesh, which decimates to itself
  owned_t<TestType> nothing;
  CHECK(tf::cpp::decimated(nothing.mesh(), TestType{0.5}).mesh.size() == 0);

  auto source = box(TestType{2});
  auto wrong_size = filled_array<std::int32_t>(source.polygons.size() - 1, 0);
  auto wrong_rank =
      filled_array<std::int32_t>(source.polygons.size(), 0)
          .reshape({2, static_cast<int>(source.polygons.size() / 2)});
  CHECK_THROWS_AS(
      tf::cpp::decimated(source.mesh(), TestType{0.5}, {}, wrong_size),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::simplified(source.mesh(), {}, wrong_rank),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::decimated(source.mesh(), TestType{-0.1}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::isotropic_remeshed(
          source.mesh(), tf::isotropic_remesh_config<TestType>(TestType{0})),
      std::invalid_argument);
  tf::simplify_config<TestType> bad_simplify;
  bad_simplify.lambda = std::numeric_limits<TestType>::quiet_NaN();
  CHECK_THROWS_AS(tf::cpp::simplified(source.mesh(), bad_simplify),
                  std::invalid_argument);

  // the cache owns the indices for the reading it answers, so a read refuses
  // what a restatement of the points put out of reach
  source.polygons.points_buffer().data_buffer().allocate(0);
  source.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::decimated(source.mesh(), TestType{0.5}),
                  std::out_of_range);
}

TEMPLATE_TEST_CASE("remesh async retains its regions and submits once",
                   "[cpp][remesh][async]", float, double) {
  auto source = box(TestType{2});
  auto regions = filled_array<std::int32_t>(source.polygons.size(), 7);
  tf::decimate_config<TestType> config;
  config.parallel = false;

  auto pending =
      tf::cpp::async::decimated(source.mesh(), TestType{1}, config, regions);
  static_assert(std::is_same_v<decltype(pending),
                               std::future<tf::cpp::remesh_result<
                                   tf::cpp::default_index_t, TestType>>>);
  static_assert(
      std::is_same_v<decltype(tf::cpp::async::isotropic_remeshed(
                         source.mesh(),
                         tf::isotropic_remesh_config<TestType>(TestType{1}))),
                     std::future<tf::cpp::remesh_result<
                         tf::cpp::default_index_t, TestType>>>);
  static_assert(
      std::is_same_v<decltype(tf::cpp::async::simplified(source.mesh())),
                     std::future<tf::cpp::remesh_result<
                         tf::cpp::default_index_t, TestType>>>);

  regions.destroy();
  auto result = pending.get();
  CHECK(result.regions[0] == 7);

  auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom_source = box(TestType{2});
  auto custom = tf::cpp::async::decimated(counting_resolver{submissions},
                                          custom_source.mesh(), TestType{1});
  static_assert(std::is_same_v<decltype(custom),
                               std::future<tf::cpp::remesh_result<
                                   tf::cpp::default_index_t, TestType>>>);
  CHECK(custom.get().mesh.size() == custom_source.polygons.size());
  CHECK(submissions->load(std::memory_order_relaxed) == 1);

  auto failure = tf::cpp::async::decimated(custom_source.mesh(), TestType{2});
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "remesh review resolver retains isotropic and simplified regions once",
    "[cpp][remesh][review][async]", float, double) {
  {
    auto source = box(TestType{2});
    auto regions = filled_array<std::int32_t>(source.polygons.size(), 11);
    auto submissions = std::make_shared<std::atomic<int>>(0);
    tf::isotropic_remesh_config<TestType> config(TestType{1}, 0);
    config.parallel = false;

    auto pending = tf::cpp::async::isotropic_remeshed(
        counting_resolver{submissions}, source.mesh(), config, regions);
    regions.destroy();

    const auto result = pending.get();
    CHECK(submissions->load(std::memory_order_relaxed) == 1);
    REQUIRE_FALSE(result.regions.empty());
    CHECK(result.regions[0] == 11);
  }

  {
    auto source = box(TestType{2});
    auto regions = filled_array<std::int32_t>(source.polygons.size(), 13);
    auto submissions = std::make_shared<std::atomic<int>>(0);
    tf::simplify_config<TestType> config;
    config.iterations = 0;
    config.parallel = false;

    auto pending = tf::cpp::async::simplified(counting_resolver{submissions},
                                              source.mesh(), config, regions);
    regions.destroy();

    const auto result = pending.get();
    CHECK(submissions->load(std::memory_order_relaxed) == 1);
    REQUIRE_FALSE(result.regions.empty());
    CHECK(result.regions[0] == 13);
  }
}

TEMPLATE_TEST_CASE("remesh review validates remaining numeric options",
                   "[cpp][remesh][review][validation]", float, double) {
  auto source = box(TestType{2});

  tf::decimate_config<TestType> decimate;
  decimate.feature_angle =
      tf::rad<TestType>(std::numeric_limits<TestType>::infinity());
  CHECK_THROWS_AS(tf::cpp::decimated(source.mesh(), TestType{0.5}, decimate),
                  std::invalid_argument);
  decimate = {};
  decimate.feature_weight = TestType{-1};
  CHECK_THROWS_AS(tf::cpp::decimated(source.mesh(), TestType{0.5}, decimate),
                  std::invalid_argument);

  tf::isotropic_remesh_config<TestType> isotropic(TestType{1});
  isotropic.iterations = -1;
  CHECK_THROWS_AS(tf::cpp::isotropic_remeshed(source.mesh(), isotropic),
                  std::invalid_argument);
  isotropic.iterations = 1;
  isotropic.relaxation_iters = -1;
  CHECK_THROWS_AS(tf::cpp::isotropic_remeshed(source.mesh(), isotropic),
                  std::invalid_argument);
  isotropic.relaxation_iters = 1;
  isotropic.lambda = TestType{0};
  CHECK_THROWS_AS(tf::cpp::isotropic_remeshed(source.mesh(), isotropic),
                  std::invalid_argument);

  tf::simplify_config<TestType> simplify;
  simplify.error_rel = TestType{-1};
  CHECK_THROWS_AS(tf::cpp::simplified(source.mesh(), simplify),
                  std::invalid_argument);
  simplify = {};
  simplify.optimize_iterations = -1;
  CHECK_THROWS_AS(tf::cpp::simplified(source.mesh(), simplify),
                  std::invalid_argument);
  simplify.optimize_iterations = 1;
  simplify.stabilizer = -1;
  CHECK_THROWS_AS(tf::cpp::simplified(source.mesh(), simplify),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE("remesh review latest uses a non-isometric operation frame",
                   "[cpp][remesh][review-latest][transform]", float, double) {
  auto local = subdivided_box(TestType{2});
  auto transformed = subdivided_box(TestType{2});
  transformed.place(scaling(TestType{4}));

  tf::isotropic_remesh_config<TestType> config(TestType{1}, 2);
  config.parallel = false;
  config.preserve_boundary = false;
  const auto local_result = tf::cpp::isotropic_remeshed(local.mesh(), config);
  const auto transformed_result =
      tf::cpp::isotropic_remeshed(transformed.mesh(), config);

  CHECK(transformed_result.mesh.size() != local_result.mesh.size());
  CHECK(transformed_result.mesh.points_buffer().size() !=
        local_result.mesh.points_buffer().size());

  TestType maximum_x{};
  for (const auto point : transformed_result.mesh.points())
    maximum_x = std::max(maximum_x, std::abs(point[0]));
  CHECK(maximum_x < TestType{2});
}
