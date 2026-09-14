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

#include "trueform/cpp/core/build_tree.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/spatial/async/ray_cast.hpp"
#include "trueform/cpp/spatial/ray_cast.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

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

template <typename T>
auto make_empty(tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(0);
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real>
auto ray(Real x = Real{0}, Real y = Real{0}, Real z = Real{-1})
    -> tf::cpp::primitive<Real> {
  return tf::cpp::primitive<Real>(
      tf::cpp::primitive_kind::ray,
      make_array<Real>({x, y, z, Real{0}, Real{0}, Real{1}}, {2, 3}));
}

template <typename Real> auto rays() -> tf::cpp::primitive<Real> {
  return tf::cpp::primitive<Real>(
      tf::cpp::primitive_kind::ray,
      make_array<Real>({Real{0}, Real{0}, Real{-1}, Real{0}, Real{0}, Real{1},
                        Real{0}, Real{0}, Real{-2}, Real{0}, Real{0}, Real{1}},
                       {2, 2, 3}));
}

template <typename Real>
auto triangle(Real z = Real{0}) -> tf::cpp::primitive<Real> {
  return tf::cpp::primitive<Real>(
      tf::cpp::primitive_kind::triangle,
      make_array<Real>(
          {Real{-1}, Real{-1}, z, Real{1}, Real{-1}, z, Real{0}, Real{1}, z},
          {3, 3}));
}

template <typename Real> auto triangles() -> tf::cpp::primitive<Real> {
  return tf::cpp::primitive<Real>(
      tf::cpp::primitive_kind::triangle,
      make_array<Real>({Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0},
                        Real{0}, Real{1}, Real{0}, Real{-1}, Real{-1}, Real{2},
                        Real{1}, Real{-1}, Real{2}, Real{0}, Real{1}, Real{2}},
                       {2, 3, 3}));
}

template <typename Real>
auto triangle_mesh()
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0},
                  Real{0}, Real{1}, Real{0}})};
}

/// What a caller assembles when no handle of its own outlives the call: the
/// storage lives in a shared block and the mesh retains it.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto held_carrier_of(tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon> owner)
    -> tf::cpp::mesh<Index, Real, Dims, Ngon> {
  const auto held = std::make_shared<
      const tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon>>(
      std::move(owner));
  return {held->polygons.faces(), held->polygons.points(), held->cache,
          held->frame(), held};
}

template <typename Real>
auto origin_cloud() -> tf::cpp::test::owned_point_cloud<Real> {
  return {tf::cpp::test::points_of<Real>({Real{0}, Real{0}, Real{0}})};
}

/// The placement a carrier is assembled at, in the shape `place` states it.
template <typename Real> auto translation(Real z) -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, Real{0}, Real{0}, Real{1},
          Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, z,
          Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Real>
auto target(tf::cpp::primitive_kind kind) -> tf::cpp::primitive<Real> {
  switch (kind) {
  case tf::cpp::primitive_kind::point:
    return tf::cpp::primitive<Real>(kind, make_array<Real>({0, 0, 0}, {3}));
  case tf::cpp::primitive_kind::segment:
    return tf::cpp::primitive<Real>(
        kind, make_array<Real>({-1, 0, 0, 1, 0, 0}, {2, 3}));
  case tf::cpp::primitive_kind::triangle:
    return triangle<Real>();
  case tf::cpp::primitive_kind::ray:
  case tf::cpp::primitive_kind::line:
    return tf::cpp::primitive<Real>(
        kind, make_array<Real>({0, 0, 0, 1, 0, 0}, {2, 3}));
  case tf::cpp::primitive_kind::plane:
    return tf::cpp::primitive<Real>(kind, make_array<Real>({0, 0, 1, 0}, {4}));
  case tf::cpp::primitive_kind::aabb:
    return tf::cpp::primitive<Real>(
        kind, make_array<Real>({-1, -1, -1, 1, 1, 1}, {2, 3}));
  case tf::cpp::primitive_kind::polygon:
    return tf::cpp::primitive<Real>(
        kind,
        make_array<Real>({-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0}, {4, 3}));
  case tf::cpp::primitive_kind::vector:
    return tf::cpp::primitive<Real>(kind, make_array<Real>({0, 0, 1}, {3}));
  }
  throw std::logic_error("unknown primitive kind");
}

template <typename Real>
auto ray_2d(Real x = Real{-1}, Real y = Real{0}, Real dx = Real{1},
            Real dy = Real{0}) -> tf::cpp::primitive<Real, 2> {
  return tf::cpp::primitive<Real, 2>(tf::cpp::primitive_kind::ray,
                                     make_array<Real>({x, y, dx, dy}, {2, 2}));
}

template <typename Real> auto rays_2d() -> tf::cpp::primitive<Real, 2> {
  return tf::cpp::primitive<Real, 2>(
      tf::cpp::primitive_kind::ray,
      make_array<Real>({Real{-1}, Real{0.5}, Real{1}, Real{0}, Real{2},
                        Real{0.5}, Real{-1}, Real{0}},
                       {2, 2, 2}));
}

template <typename Real>
auto target_2d(tf::cpp::primitive_kind kind) -> tf::cpp::primitive<Real, 2> {
  using primitive_type = tf::cpp::primitive<Real, 2>;
  switch (kind) {
  case tf::cpp::primitive_kind::point:
    return primitive_type(kind, make_array<Real>({0, 0}, {2}));
  case tf::cpp::primitive_kind::segment:
    return primitive_type(kind, make_array<Real>({0, -1, 0, 1}, {2, 2}));
  case tf::cpp::primitive_kind::triangle:
    return primitive_type(kind, make_array<Real>({0, -1, 1, 0, 0, 1}, {3, 2}));
  case tf::cpp::primitive_kind::ray:
  case tf::cpp::primitive_kind::line:
    return primitive_type(kind, make_array<Real>({0, -1, 0, 1}, {2, 2}));
  case tf::cpp::primitive_kind::aabb:
    return primitive_type(kind, make_array<Real>({0, -1, 1, 1}, {2, 2}));
  case tf::cpp::primitive_kind::polygon:
    return primitive_type(kind,
                          make_array<Real>({0, -1, 1, -1, 1, 1, 0, 1}, {4, 2}));
  case tf::cpp::primitive_kind::vector:
    return primitive_type(kind, make_array<Real>({1, 0}, {2}));
  case tf::cpp::primitive_kind::plane:
    return primitive_type(kind, make_array<Real>({0, 1, 0}, {3}));
  }
  throw std::logic_error("unknown 2D primitive kind");
}

template <typename Real>
auto empty_target_2d(tf::cpp::primitive_kind kind)
    -> tf::cpp::primitive<Real, 2> {
  using primitive_type = tf::cpp::primitive<Real, 2>;
  switch (kind) {
  case tf::cpp::primitive_kind::point:
    return primitive_type(kind, make_empty<Real>({0, 2}));
  case tf::cpp::primitive_kind::segment:
  case tf::cpp::primitive_kind::ray:
  case tf::cpp::primitive_kind::line:
  case tf::cpp::primitive_kind::aabb:
    return primitive_type(kind, make_empty<Real>({0, 2, 2}));
  case tf::cpp::primitive_kind::triangle:
    return primitive_type(kind, make_empty<Real>({0, 3, 2}));
  case tf::cpp::primitive_kind::polygon:
    return primitive_type(kind, make_empty<Real>({0, 4, 2}));
  case tf::cpp::primitive_kind::vector:
  case tf::cpp::primitive_kind::plane:
    break;
  }
  throw std::logic_error("unsupported empty 2D primitive kind");
}

template <typename Rays, typename Target, typename = void>
struct has_ray_cast : std::false_type {};

template <typename Rays, typename Target>
struct has_ray_cast<
    Rays, Target,
    std::void_t<decltype(tf::cpp::ray_cast(std::declval<const Rays &>(),
                                           std::declval<const Target &>()))>>
    : std::true_type {};

template <typename Rays, typename Target, typename = void>
struct has_async_ray_cast : std::false_type {};

template <typename Rays, typename Target>
struct has_async_ray_cast<
    Rays, Target,
    std::void_t<decltype(tf::cpp::async::ray_cast(
        std::declval<const Rays &>(), std::declval<const Target &>()))>>
    : std::true_type {};

template <typename RayReal, typename TargetReal>
using ray_cast_2d_function = tf::cpp::ray_cast_primitive_result<
    std::common_type_t<RayReal, TargetReal>> (*)(
    const tf::cpp::primitive<RayReal, 2> &,
    const tf::cpp::primitive<TargetReal, 2> &,
    const tf::cpp::ray_cast_options<std::common_type_t<RayReal, TargetReal>> &);

} // namespace

TEMPLATE_TEST_CASE("2D ray cast supports every valid primitive target kind",
                   "[cpp][spatial][ray_cast][2d][primitive]", float, double) {
  constexpr std::array<tf::cpp::primitive_kind, 7> kinds{
      tf::cpp::primitive_kind::point,    tf::cpp::primitive_kind::segment,
      tf::cpp::primitive_kind::triangle, tf::cpp::primitive_kind::ray,
      tf::cpp::primitive_kind::line,     tf::cpp::primitive_kind::aabb,
      tf::cpp::primitive_kind::polygon};
  constexpr std::array<std::size_t, 7> strides{2, 4, 6, 4, 4, 4, 8};
  const auto query = ray_2d<TestType>();
  REQUIRE(query.element_stride() == 4);

  for (std::size_t index = 0; index < kinds.size(); ++index) {
    INFO("target kind " << static_cast<int>(kinds[index]));
    const auto geometry = target_2d<TestType>(kinds[index]);
    CHECK(geometry.element_stride() == strides[index]);
    const auto result = tf::cpp::ray_cast(query, geometry);
    REQUIRE(result.is_scalar());
    CHECK(result.scalar().hit);
    CHECK(result.scalar().t == TestType{1});
    CHECK(result.scalar().element_id == -1);
  }

  const auto vector = target_2d<TestType>(tf::cpp::primitive_kind::vector);
  CHECK_THROWS_AS(tf::cpp::ray_cast(query, vector), std::invalid_argument);
  CHECK_THROWS_AS(target_2d<TestType>(tf::cpp::primitive_kind::plane),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "2D primitive ray cast preserves scalar broadcast pairwise and empty modes",
    "[cpp][spatial][ray_cast][2d][batch]", float, double) {
  const auto one_ray = ray_2d<TestType>();
  const auto ray_batch = rays_2d<TestType>();
  const auto one_target = target_2d<TestType>(tf::cpp::primitive_kind::segment);
  const auto target_batch = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::segment,
      make_array<TestType>({0, 0, 0, 1, 1, 0, 1, 1}, {2, 2, 2}));

  REQUIRE(one_ray.element_stride() == 4);
  REQUIRE(ray_batch.element_stride() == 4);
  REQUIRE(one_target.element_stride() == 4);
  REQUIRE(target_batch.element_stride() == 4);

  const auto scalar = tf::cpp::ray_cast(one_ray, one_target);
  const auto rays_broadcast = tf::cpp::ray_cast(ray_batch, one_target);
  const auto targets_broadcast = tf::cpp::ray_cast(one_ray, target_batch);
  const auto pairwise = tf::cpp::ray_cast(ray_batch, target_batch);
  REQUIRE(scalar.is_scalar());
  REQUIRE(rays_broadcast.is_batch());
  REQUIRE(targets_broadcast.is_batch());
  REQUIRE(pairwise.is_batch());
  CHECK(scalar.scalar().hit);
  CHECK(scalar.scalar().t == TestType{1});
  CHECK(rays_broadcast.batch().hits.raw_shape() == tf::small_vector<int, 3>{2});
  CHECK(rays_broadcast.batch().hits[0] == 1);
  CHECK(rays_broadcast.batch().hits[1] == 1);
  CHECK(rays_broadcast.batch().ts[0] == TestType{1});
  CHECK(rays_broadcast.batch().ts[1] == TestType{2});
  CHECK(targets_broadcast.batch().hits[0] == 1);
  CHECK(targets_broadcast.batch().hits[1] == 1);
  CHECK(targets_broadcast.batch().ts[0] == TestType{1});
  CHECK(targets_broadcast.batch().ts[1] == TestType{2});
  CHECK(pairwise.batch().hits[0] == 1);
  CHECK(pairwise.batch().hits[1] == 1);
  CHECK(pairwise.batch().ts[0] == TestType{1});
  CHECK(pairwise.batch().ts[1] == TestType{1});

  const auto unequal_targets = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::point,
      make_array<TestType>({0, 0, 1, 0, 2, 0}, {3, 2}));
  CHECK_THROWS_AS(tf::cpp::ray_cast(ray_batch, unequal_targets),
                  std::invalid_argument);

  const auto empty_rays = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::ray, make_empty<TestType>({0, 2, 2}));
  REQUIRE(empty_rays.element_stride() == 4);
  constexpr std::array<tf::cpp::primitive_kind, 7> kinds{
      tf::cpp::primitive_kind::point,    tf::cpp::primitive_kind::segment,
      tf::cpp::primitive_kind::triangle, tf::cpp::primitive_kind::ray,
      tf::cpp::primitive_kind::line,     tf::cpp::primitive_kind::aabb,
      tf::cpp::primitive_kind::polygon};
  constexpr std::array<std::size_t, 7> strides{2, 4, 6, 4, 4, 4, 8};
  for (std::size_t index = 0; index < kinds.size(); ++index) {
    INFO("empty target kind " << static_cast<int>(kinds[index]));
    const auto empty_target = empty_target_2d<TestType>(kinds[index]);
    CHECK(empty_target.element_stride() == strides[index]);
    const auto result = tf::cpp::ray_cast(empty_rays, empty_target);
    REQUIRE(result.is_batch());
    CHECK(result.batch().hits.raw_shape() == tf::small_vector<int, 3>{0});
    CHECK(result.batch().ts.raw_shape() == tf::small_vector<int, 3>{0});
    CHECK(result.batch().hits.empty());
    CHECK(result.batch().ts.empty());
  }
}

TEMPLATE_TEST_CASE("2D ray cast applies inclusive scalar and batch bounds",
                   "[cpp][spatial][ray_cast][2d][bounds]", float, double) {
  const auto query = rays_2d<TestType>();
  const auto geometry = target_2d<TestType>(tf::cpp::primitive_kind::segment);

  tf::cpp::ray_cast_options<TestType> mixed;
  mixed.min_t = TestType{0};
  mixed.max_ts = make_array<TestType>({TestType{0.5}, TestType{2}}, {2});
  const auto mixed_result = tf::cpp::ray_cast(query, geometry, mixed).batch();
  CHECK(mixed_result.hits[0] == 0);
  CHECK(mixed_result.ts[0] == TestType{1});
  CHECK(mixed_result.hits[1] == 1);
  CHECK(mixed_result.ts[1] == TestType{2});

  tf::cpp::ray_cast_options<TestType> inclusive;
  inclusive.min_ts = make_array<TestType>({TestType{1}, TestType{2}}, {2});
  inclusive.max_t = TestType{2};
  const auto inclusive_result =
      tf::cpp::ray_cast(query, geometry, inclusive).batch();
  CHECK(inclusive_result.hits[0] == 1);
  CHECK(inclusive_result.hits[1] == 1);

  tf::cpp::ray_cast_options<TestType> exact;
  exact.min_t = TestType{1};
  exact.max_t = TestType{1};
  const auto exact_result =
      tf::cpp::ray_cast(query.at(0), geometry, exact).scalar();
  CHECK(exact_result.hit);
  CHECK(exact_result.t == TestType{1});

  tf::cpp::ray_cast_options<TestType> inverted;
  inverted.min_t = TestType{2};
  inverted.max_t = TestType{1};
  CHECK_FALSE(tf::cpp::ray_cast(query.at(0), geometry, inverted).scalar().hit);

  tf::cpp::ray_cast_options<TestType> wrong_count;
  wrong_count.min_ts = make_array<TestType>({TestType{0}}, {1});
  CHECK_THROWS_AS(tf::cpp::ray_cast(query, geometry, wrong_count),
                  std::invalid_argument);
  tf::cpp::ray_cast_options<TestType> wrong_rank;
  wrong_rank.max_ts = make_array<TestType>({0, 2}, {1, 2});
  CHECK_THROWS_AS(tf::cpp::ray_cast(query, geometry, wrong_rank),
                  std::invalid_argument);
}

TEST_CASE("2D ray cast rejects mismatched dimensions and form overloads",
          "[cpp][spatial][ray_cast][2d][compile-time]") {
  using ray2 = tf::cpp::primitive<float, 2>;
  using ray3 = tf::cpp::primitive<float, 3>;
  using target2 = tf::cpp::primitive<double, 2>;
  using target3 = tf::cpp::primitive<double, 3>;

  STATIC_REQUIRE(has_ray_cast<ray2, target2>::value);
  STATIC_REQUIRE(has_async_ray_cast<ray2, target2>::value);
  STATIC_REQUIRE_FALSE(has_ray_cast<ray2, target3>::value);
  STATIC_REQUIRE_FALSE(has_ray_cast<ray3, target2>::value);
  STATIC_REQUIRE_FALSE(has_async_ray_cast<ray2, target3>::value);
  STATIC_REQUIRE_FALSE(has_async_ray_cast<ray3, target2>::value);

  STATIC_REQUIRE_FALSE(
      has_ray_cast<ray2,
                   tf::cpp::mesh<tf::cpp::default_index_t, float>>::value);
  STATIC_REQUIRE_FALSE(has_ray_cast<ray2, tf::cpp::point_cloud<float>>::value);
  STATIC_REQUIRE_FALSE(
      has_async_ray_cast<
          ray2, tf::cpp::mesh<tf::cpp::default_index_t, float>>::value);
  STATIC_REQUIRE_FALSE(
      has_async_ray_cast<ray2, tf::cpp::point_cloud<float>>::value);
  STATIC_REQUIRE(
      has_ray_cast<ray2, tf::cpp::mesh<std::int32_t, float, 2>>::value);
  STATIC_REQUIRE(has_ray_cast<ray2, tf::cpp::point_cloud<float, 2>>::value);
  STATIC_REQUIRE(
      has_ray_cast<ray2, tf::cpp::edge_mesh<std::int64_t, float, 2>>::value);
  STATIC_REQUIRE(
      has_async_ray_cast<ray2, tf::cpp::mesh<std::int32_t, float, 2>>::value);
  STATIC_REQUIRE(
      has_async_ray_cast<ray2, tf::cpp::point_cloud<float, 2>>::value);
  STATIC_REQUIRE(
      has_async_ray_cast<ray2,
                         tf::cpp::edge_mesh<std::int64_t, float, 2>>::value);

  STATIC_REQUIRE(has_ray_cast<ray3, target3>::value);
  STATIC_REQUIRE(has_async_ray_cast<ray3, target3>::value);
  STATIC_REQUIRE(
      has_ray_cast<ray3,
                   tf::cpp::mesh<tf::cpp::default_index_t, float>>::value);
  STATIC_REQUIRE(has_ray_cast<ray3, tf::cpp::point_cloud<float>>::value);
  STATIC_REQUIRE(has_async_ray_cast<
                 ray3, tf::cpp::mesh<tf::cpp::default_index_t, float>>::value);
  STATIC_REQUIRE(has_async_ray_cast<ray3, tf::cpp::point_cloud<float>>::value);
}

TEST_CASE("2D ray cast precision orders link exact archive symbols",
          "[cpp][spatial][ray_cast][2d][precision][archive-link]") {
  const ray_cast_2d_function<float, float> float_float =
      &tf::cpp::ray_cast<float, float, 2>;
  const ray_cast_2d_function<double, double> double_double =
      &tf::cpp::ray_cast<double, double, 2>;
  const ray_cast_2d_function<float, double> float_double =
      &tf::cpp::ray_cast<float, double, 2>;
  const ray_cast_2d_function<double, float> double_float =
      &tf::cpp::ray_cast<double, float, 2>;

  const auto ray32 = ray_2d<float>();
  const auto ray64 = ray_2d<double>();
  const auto target32 = target_2d<float>(tf::cpp::primitive_kind::segment);
  const auto target64 = target_2d<double>(tf::cpp::primitive_kind::segment);
  CHECK(float_float(ray32, target32, {}).scalar().hit);
  CHECK(double_double(ray64, target64, {}).scalar().hit);
  CHECK(float_double(ray32, target64, {}).scalar().hit);
  CHECK(double_float(ray64, target32, {}).scalar().hit);

  const auto widened0 = tf::cpp::ray_cast(ray32, target64);
  const auto widened1 = tf::cpp::ray_cast(ray64, target32);
  STATIC_REQUIRE(
      std::is_same_v<decltype(widened0),
                     const tf::cpp::ray_cast_primitive_result<double>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(widened1),
                     const tf::cpp::ray_cast_primitive_result<double>>);
  CHECK(widened0.scalar().t == 1.0);
  CHECK(widened1.scalar().t == 1.0);
}

TEMPLATE_TEST_CASE(
    "async 2D ray cast preserves exact futures resolver ownership and errors",
    "[cpp][spatial][ray_cast][2d][async]", float, double) {
  auto future = [] {
    const auto query = rays_2d<TestType>();
    const auto geometry = tf::cpp::primitive<TestType, 2>(
        tf::cpp::primitive_kind::segment,
        make_array<TestType>({0, 0, 0, 1, 1, 0, 1, 1}, {2, 2, 2}));
    tf::cpp::ray_cast_options<TestType> options;
    options.min_ts = make_array<TestType>({TestType{1}, TestType{1}}, {2});
    options.max_t = TestType{1};
    return tf::cpp::async::ray_cast(query, geometry, options);
  }();
  STATIC_REQUIRE(std::is_same_v<
                 decltype(future),
                 std::future<tf::cpp::ray_cast_primitive_result<TestType>>>);
  const auto owned = future.get();
  REQUIRE(owned.is_batch());
  CHECK(owned.batch().hits[0] == 1);
  CHECK(owned.batch().hits[1] == 1);
  CHECK(owned.batch().ts[0] == TestType{1});
  CHECK(owned.batch().ts[1] == TestType{1});

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::ray_cast(
      counting_resolver{submissions}, ray_2d<TestType>(),
      target_2d<TestType>(tf::cpp::primitive_kind::segment));
  STATIC_REQUIRE(std::is_same_v<
                 decltype(custom),
                 std::future<tf::cpp::ray_cast_primitive_result<TestType>>>);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  const auto custom_result = custom.get().scalar();
  CHECK(custom_result.hit);
  CHECK(custom_result.t == TestType{1});

  const auto unequal = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::point,
      make_array<TestType>({0, 0, 1, 0, 2, 0}, {3, 2}));
  auto failure = tf::cpp::async::ray_cast(rays_2d<TestType>(), unequal);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);

  const auto vector = target_2d<TestType>(tf::cpp::primitive_kind::vector);
  auto vector_failure = tf::cpp::async::ray_cast(ray_2d<TestType>(), vector);
  CHECK_THROWS_AS(vector_failure.get(), std::invalid_argument);
}

TEST_CASE("async 2D ray cast preserves both mixed precision orders",
          "[cpp][spatial][ray_cast][2d][async][mixed]") {
  auto float_double = tf::cpp::async::ray_cast(
      ray_2d<float>(), target_2d<double>(tf::cpp::primitive_kind::segment));
  auto double_float = tf::cpp::async::ray_cast(
      ray_2d<double>(), target_2d<float>(tf::cpp::primitive_kind::segment));
  STATIC_REQUIRE(
      std::is_same_v<decltype(float_double),
                     std::future<tf::cpp::ray_cast_primitive_result<double>>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(double_float),
                     std::future<tf::cpp::ray_cast_primitive_result<double>>>);
  const auto float_double_result = float_double.get().scalar();
  const auto double_float_result = double_float.get().scalar();
  CHECK(float_double_result.hit);
  CHECK(float_double_result.t == 1.0);
  CHECK(double_float_result.hit);
  CHECK(double_float_result.t == 1.0);
}

TEMPLATE_TEST_CASE("ray cast supports every primitive target kind",
                   "[cpp][spatial][ray_cast][primitive]", float, double) {
  const tf::cpp::primitive_kind kinds[]{
      tf::cpp::primitive_kind::point,    tf::cpp::primitive_kind::segment,
      tf::cpp::primitive_kind::triangle, tf::cpp::primitive_kind::ray,
      tf::cpp::primitive_kind::line,     tf::cpp::primitive_kind::plane,
      tf::cpp::primitive_kind::aabb,     tf::cpp::primitive_kind::polygon};
  const auto query = ray<TestType>();
  for (const auto kind : kinds) {
    INFO("target kind " << static_cast<int>(kind));
    const auto result = tf::cpp::ray_cast(query, target<TestType>(kind));
    REQUIRE(result.is_scalar());
    CHECK(result.scalar().hit);
    CHECK(result.scalar().element_id == -1);
  }
}

TEMPLATE_TEST_CASE("primitive ray cast broadcasts and pairs batches",
                   "[cpp][spatial][ray_cast][batch]", float, double) {
  const auto one_ray = ray<TestType>();
  const auto ray_batch = rays<TestType>();
  const auto one_target = triangle<TestType>();
  const auto target_batch = triangles<TestType>();

  const auto scalar = tf::cpp::ray_cast(one_ray, one_target);
  const auto rays_broadcast = tf::cpp::ray_cast(ray_batch, one_target);
  const auto targets_broadcast = tf::cpp::ray_cast(one_ray, target_batch);
  const auto pairwise = tf::cpp::ray_cast(ray_batch, target_batch);
  CHECK(scalar.is_scalar());
  REQUIRE(rays_broadcast.is_batch());
  REQUIRE(targets_broadcast.is_batch());
  REQUIRE(pairwise.is_batch());
  CHECK(rays_broadcast.batch().hits.raw_shape() == tf::small_vector<int, 3>{2});
  CHECK(rays_broadcast.batch().ts[0] == Catch::Approx(1));
  CHECK(rays_broadcast.batch().ts[1] == Catch::Approx(2));
  CHECK(targets_broadcast.batch().ts[0] == Catch::Approx(1));
  CHECK(targets_broadcast.batch().ts[1] == Catch::Approx(3));
  CHECK(pairwise.batch().ts[0] == Catch::Approx(1));
  CHECK(pairwise.batch().ts[1] == Catch::Approx(4));
}

TEMPLATE_TEST_CASE("ray cast bounds mix scalar and per-result policies",
                   "[cpp][spatial][ray_cast][bounds]", float, double) {
  const auto query = rays<TestType>();
  const auto geometry = triangle<TestType>();

  tf::cpp::ray_cast_options<TestType> scalar_min;
  scalar_min.min_t = 0;
  scalar_min.max_ts = make_array<TestType>({TestType{0.5}, TestType{3}}, {2});
  const auto a = tf::cpp::ray_cast(query, geometry, scalar_min).batch();
  CHECK(a.hits[0] == 0);
  CHECK(a.hits[1] == 1);

  tf::cpp::ray_cast_options<TestType> scalar_max;
  scalar_max.min_ts = make_array<TestType>({TestType{2}, TestType{0}}, {2});
  scalar_max.max_t = 3;
  const auto b = tf::cpp::ray_cast(query, geometry, scalar_max).batch();
  CHECK(b.hits[0] == 0);
  CHECK(b.hits[1] == 1);

  tf::cpp::ray_cast_options<TestType> inverted;
  inverted.min_t = 3;
  inverted.max_t = 1;
  CHECK_FALSE(
      tf::cpp::ray_cast(ray<TestType>(), geometry, inverted).scalar().hit);
}

TEMPLATE_TEST_CASE("ray cast preserves miss t and ID defaults",
                   "[cpp][spatial][ray_cast][miss]", float, double) {
  auto plane = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::plane, make_array<TestType>({0, 0, 1, -4}, {4}));
  tf::cpp::ray_cast_options<TestType> bounded;
  bounded.max_t = 3;
  const auto primitive_miss =
      tf::cpp::ray_cast(ray<TestType>(), plane, bounded);
  REQUIRE_FALSE(primitive_miss.scalar().hit);
  CHECK(primitive_miss.scalar().t == Catch::Approx(5));
  CHECK(primitive_miss.scalar().element_id == -1);

  const auto mesh_storage = triangle_mesh<TestType>();
  const auto form_miss =
      tf::cpp::ray_cast(ray<TestType>(5, 5), mesh_storage.mesh());
  REQUIRE_FALSE(form_miss.scalar().hit);
  CHECK(form_miss.scalar().t == TestType{0});
  CHECK(form_miss.scalar().element_id == -1);
}

TEMPLATE_TEST_CASE("near-epsilon inverted ray bounds always miss",
                   "[cpp][spatial][ray_cast][bounds][inverted]", float,
                   double) {
  const auto inverted_min =
      std::nextafter(TestType{1}, std::numeric_limits<TestType>::max());
  REQUIRE(inverted_min > TestType{1});
  auto geometry = triangle<TestType>();
  const auto mesh_storage = triangle_mesh<TestType>();
  const auto mesh = mesh_storage.mesh();

  tf::cpp::ray_cast_options<TestType> scalar_options;
  scalar_options.min_t = inverted_min;
  scalar_options.max_t = TestType{1};
  const auto primitive_scalar =
      tf::cpp::ray_cast(ray<TestType>(), geometry, scalar_options).scalar();
  const auto form_scalar =
      tf::cpp::ray_cast(ray<TestType>(), mesh, scalar_options).scalar();
  CHECK_FALSE(primitive_scalar.hit);
  CHECK(primitive_scalar.t == Catch::Approx(1));
  CHECK(primitive_scalar.element_id == -1);
  CHECK_FALSE(form_scalar.hit);
  CHECK(form_scalar.t == TestType{0});
  CHECK(form_scalar.element_id == -1);

  auto duplicate_rays = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::ray,
      make_array<TestType>({TestType{0}, TestType{0}, TestType{-1}, TestType{0},
                            TestType{0}, TestType{1}, TestType{0}, TestType{0},
                            TestType{-1}, TestType{0}, TestType{0},
                            TestType{1}},
                           {2, 2, 3}));
  tf::cpp::ray_cast_options<TestType> batch_options;
  batch_options.min_ts =
      make_array<TestType>({inverted_min, inverted_min}, {2});
  batch_options.max_ts = make_array<TestType>({TestType{1}, TestType{1}}, {2});
  const auto primitive_batch =
      tf::cpp::ray_cast(duplicate_rays, geometry, batch_options).batch();
  const auto form_batch =
      tf::cpp::ray_cast(duplicate_rays, mesh, batch_options).batch();
  CHECK(primitive_batch.hits[0] == 0);
  CHECK(primitive_batch.hits[1] == 0);
  CHECK(primitive_batch.ts[0] == Catch::Approx(1));
  CHECK(primitive_batch.ts[1] == Catch::Approx(1));
  CHECK(form_batch.hits[0] == 0);
  CHECK(form_batch.hits[1] == 0);
  CHECK(form_batch.ts[0] == TestType{0});
  CHECK(form_batch.ts[1] == TestType{0});
  CHECK(form_batch.element_ids[0] == -1);
  CHECK(form_batch.element_ids[1] == -1);
}

TEMPLATE_TEST_CASE("mesh and point-cloud ray casts use form precision",
                   "[cpp][spatial][ray_cast][form]", float, double) {
  const auto mesh_storage = triangle_mesh<TestType>();
  const auto cloud_storage = origin_cloud<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto cloud = cloud_storage.point_cloud();
  auto query = ray<float>();
  const auto mesh_hit = tf::cpp::ray_cast(query, mesh);
  const auto cloud_hit = tf::cpp::ray_cast(query, cloud);
  static_assert(
      std::is_same<decltype(mesh_hit),
                   const tf::cpp::ray_cast_form_result<tf::cpp::default_index_t,
                                                       TestType>>::value,
      "form result follows form precision");
  REQUIRE(mesh_hit.scalar().hit);
  REQUIRE(cloud_hit.scalar().hit);
  CHECK(mesh_hit.scalar().t == Catch::Approx(1));
  CHECK(mesh_hit.scalar().element_id == 0);
  CHECK(cloud_hit.scalar().element_id == 0);

  const auto batch = tf::cpp::ray_cast(rays<double>(), mesh);
  const auto cloud_batch = tf::cpp::ray_cast(rays<double>(), cloud);
  REQUIRE(batch.is_batch());
  REQUIRE(cloud_batch.is_batch());
  CHECK(batch.batch().hits[0] == 1);
  CHECK(batch.batch().hits[1] == 1);
  CHECK(batch.batch().element_ids[0] == 0);
  CHECK(cloud_batch.batch().element_ids[0] == 0);

  tf::cpp::ray_cast_options<TestType> mixed_bounds;
  mixed_bounds.min_t = 0;
  mixed_bounds.max_ts = make_array<TestType>({TestType{0.5}, TestType{3}}, {2});
  const auto bounded = tf::cpp::ray_cast(rays<TestType>(), mesh, mixed_bounds);
  CHECK(bounded.batch().hits[0] == 0);
  CHECK(bounded.batch().hits[1] == 1);
}

TEMPLATE_TEST_CASE("form ray cast honors transformations and reuses trees",
                   "[cpp][spatial][ray_cast][tree]", float, double) {
  auto mesh_storage = triangle_mesh<TestType>();
  const auto mesh = mesh_storage.mesh();
  CHECK_FALSE(mesh_storage.cache.is_tree_built());
  CHECK(tf::cpp::ray_cast(ray<TestType>(), mesh).scalar().hit);
  CHECK(mesh_storage.cache.is_tree_fresh(mesh.geometry()));
  CHECK(mesh_storage.cache.tree_build_count() == 1);
  CHECK(tf::cpp::ray_cast(ray<TestType>(), mesh).scalar().hit);
  CHECK(mesh_storage.cache.tree_build_count() == 1);

  // a cache is built in local coordinates, so a placement is not a change
  mesh_storage.place(translation<TestType>(2));
  const auto transformed =
      tf::cpp::ray_cast(ray<TestType>(), mesh_storage.mesh());
  REQUIRE(transformed.scalar().hit);
  CHECK(transformed.scalar().t == Catch::Approx(3));
  CHECK(mesh_storage.cache.tree_build_count() == 1);
  mesh_storage.place(translation<TestType>(0));
  CHECK(tf::cpp::ray_cast(ray<TestType>(), mesh_storage.mesh()).scalar().t ==
        Catch::Approx(1));

  auto &coordinates = mesh_storage.polygons.points_buffer().data_buffer();
  coordinates[2] = 1;
  coordinates[5] = 1;
  coordinates[8] = 1;
  mesh_storage.cache.points_changed();
  const auto moved = mesh_storage.mesh();
  CHECK_FALSE(mesh_storage.cache.is_tree_fresh(moved.geometry()));
  CHECK(tf::cpp::ray_cast(ray<TestType>(), moved).scalar().t ==
        Catch::Approx(2));
  CHECK(mesh_storage.cache.tree_build_count() == 2);

  auto cloud_storage = origin_cloud<TestType>();
  const auto cloud = cloud_storage.point_cloud();
  CHECK_FALSE(cloud_storage.cache.is_tree_built());
  CHECK(tf::cpp::ray_cast(ray<TestType>(), cloud).scalar().hit);
  CHECK(cloud_storage.cache.is_tree_fresh(cloud.geometry()));
  CHECK(cloud_storage.cache.tree_build_count() == 1);
  cloud_storage.place(translation<TestType>(2));
  CHECK(tf::cpp::ray_cast(ray<TestType>(), cloud_storage.point_cloud())
            .scalar()
            .t == Catch::Approx(3));
  CHECK(cloud_storage.cache.tree_build_count() == 1);
}

TEMPLATE_TEST_CASE("ray cast validates rays batches bounds and handles",
                   "[cpp][spatial][ray_cast][validation]", float, double) {
  auto point = target<TestType>(tf::cpp::primitive_kind::point);
  auto vector = target<TestType>(tf::cpp::primitive_kind::vector);
  auto query_batch = rays<TestType>();
  auto target_batch = triangles<TestType>();
  auto three_targets = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::point,
      make_array<TestType>({0, 0, 0, 0, 0, 0, 0, 0, 0}, {3, 3}));
  CHECK_THROWS_AS(tf::cpp::ray_cast(point, point), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::ray_cast(ray<TestType>(), vector),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::ray_cast(query_batch, three_targets),
                  std::invalid_argument);

  tf::cpp::ray_cast_options<TestType> wrong_rank;
  wrong_rank.max_ts = make_array<TestType>({1, 2}, {1, 2});
  CHECK_THROWS_AS(tf::cpp::ray_cast(query_batch, point, wrong_rank),
                  std::invalid_argument);
  tf::cpp::ray_cast_options<TestType> wrong_count;
  wrong_count.min_ts = make_array<TestType>({0}, {1});
  CHECK_THROWS_AS(tf::cpp::ray_cast(query_batch, point, wrong_count),
                  std::invalid_argument);
  CHECK_NOTHROW(tf::cpp::ray_cast(query_batch, target_batch));

  // a ray meets nothing in the empty carrier
  const auto nothing_storage =
      tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType>{};
  const auto nothing_cloud_storage =
      tf::cpp::test::owned_point_cloud<TestType>{};
  CHECK(tf::cpp::ray_cast(ray<TestType>(), nothing_storage.mesh())
            .scalar()
            .element_id == -1);
  CHECK(tf::cpp::ray_cast(ray<TestType>(), nothing_cloud_storage.point_cloud())
            .scalar()
            .element_id == -1);
}

TEMPLATE_TEST_CASE("ray cast handles empty batches forms and result ownership",
                   "[cpp][spatial][ray_cast][empty]", float, double) {
  auto empty_rays = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::ray, make_empty<TestType>({0, 2, 3}));
  auto empty_targets = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::point, make_empty<TestType>({0, 3}));
  auto primitive_result = tf::cpp::ray_cast(empty_rays, empty_targets);
  REQUIRE(primitive_result.is_batch());
  CHECK(primitive_result.batch().hits.empty());
  CHECK(primitive_result.batch().ts.raw_shape() == tf::small_vector<int, 3>{0});

  const auto empty_mesh_storage =
      tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType>{};
  const auto empty_mesh = empty_mesh_storage.mesh();
  auto form_result = tf::cpp::ray_cast(empty_rays, empty_mesh);
  REQUIRE(form_result.is_batch());
  CHECK(form_result.batch().hits.empty());
  CHECK(empty_mesh_storage.cache.is_tree_fresh(empty_mesh.geometry()));

  const auto empty_cloud_storage = tf::cpp::test::owned_point_cloud<TestType>{};
  const auto empty_cloud = empty_cloud_storage.point_cloud();
  auto cloud_result = tf::cpp::ray_cast(empty_rays, empty_cloud);
  REQUIRE(cloud_result.is_batch());
  CHECK(cloud_result.batch().element_ids.empty());
  CHECK(empty_cloud_storage.cache.is_tree_fresh(empty_cloud.geometry()));

  auto owned_query = rays<TestType>();
  auto owned_target = triangle<TestType>();
  auto owned_result = tf::cpp::ray_cast(owned_query, owned_target);
  owned_query.data().destroy();
  owned_target.data().destroy();
  CHECK(owned_result.batch().hits[0] == 1);
  CHECK(owned_result.batch().ts[1] == Catch::Approx(2));

  // the result owns its own arrays, whatever becomes of the form it read
  auto owned_storage = triangle_mesh<TestType>();
  auto owned_form_result =
      tf::cpp::ray_cast(rays<TestType>(), owned_storage.mesh());
  owned_storage = {};
  CHECK(owned_form_result.batch().hits[0] == 1);
  CHECK(owned_form_result.batch().element_ids[1] == 0);
}

TEST_CASE("ray cast mixed primitive precision links archive instantiations",
          "[cpp][spatial][ray_cast][archive]") {
  const auto float_double = tf::cpp::ray_cast(ray<float>(), triangle<double>());
  const auto double_float = tf::cpp::ray_cast(ray<double>(), triangle<float>());
  static_assert(
      std::is_same<decltype(float_double),
                   const tf::cpp::ray_cast_primitive_result<double>>::value,
      "mixed primitive precision promotes to double");
  CHECK(float_double.scalar().hit);
  CHECK(double_float.scalar().hit);
}

TEMPLATE_TEST_CASE("async ray cast preserves primitive bounds and ownership",
                   "[cpp][spatial][ray_cast][async][primitive]", float,
                   double) {
  auto query = rays<TestType>();
  auto geometry = triangles<TestType>();
  tf::cpp::ray_cast_options<TestType> options;
  options.min_ts = make_array<TestType>({TestType{0}, TestType{3}}, {2});
  options.max_t = TestType{5};
  const auto expected = tf::cpp::ray_cast(query, geometry, options).batch();

  auto result = tf::cpp::async::ray_cast(query, geometry, options);
  static_assert(std::is_same_v<
                decltype(result),
                std::future<tf::cpp::ray_cast_primitive_result<TestType>>>);
  query.data().destroy();
  geometry.data().destroy();
  options.min_ts.destroy();

  const auto actual = result.get().batch();
  CHECK(actual.hits[0] == expected.hits[0]);
  CHECK(actual.hits[1] == expected.hits[1]);
  CHECK(actual.ts[0] == Catch::Approx(expected.ts[0]));
  CHECK(actual.ts[1] == Catch::Approx(expected.ts[1]));
  CHECK(actual.hits[0] == 1);
  CHECK(actual.hits[1] == 1);

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto scalar = tf::cpp::async::ray_cast(counting_resolver{submissions},
                                         ray<TestType>(), triangle<TestType>());
  static_assert(std::is_same_v<
                decltype(scalar),
                std::future<tf::cpp::ray_cast_primitive_result<TestType>>>);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(scalar.get().scalar().hit);

  tf::cpp::ray_cast_options<TestType> invalid_bounds;
  invalid_bounds.max_ts = make_array<TestType>({TestType{1}}, {1});
  auto failure = tf::cpp::async::ray_cast(rays<TestType>(),
                                          triangle<TestType>(), invalid_bounds);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}

TEST_CASE("async ray cast preserves mixed primitive precision",
          "[cpp][spatial][ray_cast][async][mixed]") {
  auto float_double =
      tf::cpp::async::ray_cast(ray<float>(), triangle<double>());
  auto double_float =
      tf::cpp::async::ray_cast(ray<double>(), triangle<float>());
  static_assert(
      std::is_same_v<decltype(float_double),
                     std::future<tf::cpp::ray_cast_primitive_result<double>>>);
  static_assert(
      std::is_same_v<decltype(double_float),
                     std::future<tf::cpp::ray_cast_primitive_result<double>>>);
  CHECK(float_double.get().scalar().hit);
  CHECK(double_float.get().scalar().hit);
}

TEMPLATE_TEST_CASE("async ray cast preserves forms caches and bound policy",
                   "[cpp][spatial][ray_cast][async][form]", float, double) {
  const auto mesh_storage = triangle_mesh<TestType>();
  const auto cloud_storage = origin_cloud<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto cloud = cloud_storage.point_cloud();
  auto query = rays<double>();
  tf::cpp::ray_cast_options<TestType> options;
  options.min_t = TestType{0};
  options.max_ts = make_array<TestType>({TestType{0.5}, TestType{3}}, {2});

  // a cache shared by concurrent jobs is filled before it is shared
  tf::cpp::build_tree(mesh);
  tf::cpp::build_tree(cloud);

  auto mesh_result = tf::cpp::async::ray_cast(query, mesh, options);
  auto cloud_result = tf::cpp::async::ray_cast(query, cloud, options);
  static_assert(std::is_same_v<decltype(mesh_result),
                               std::future<tf::cpp::ray_cast_form_result<
                                   tf::cpp::default_index_t, TestType>>>);
  static_assert(std::is_same_v<decltype(cloud_result),
                               std::future<tf::cpp::ray_cast_form_result<
                                   tf::cpp::default_index_t, TestType>>>);

  query.data().destroy();
  options.max_ts.destroy();
  const auto mesh_batch = mesh_result.get().batch();
  const auto cloud_batch = cloud_result.get().batch();
  CHECK(mesh_batch.hits[0] == 0);
  CHECK(mesh_batch.hits[1] == 1);
  CHECK(cloud_batch.hits[0] == 0);
  CHECK(cloud_batch.hits[1] == 1);
  CHECK(mesh_storage.cache.is_tree_fresh(mesh.geometry()));
  CHECK(cloud_storage.cache.is_tree_fresh(cloud.geometry()));
  CHECK(mesh_storage.cache.tree_build_count() == 1);
  CHECK(cloud_storage.cache.tree_build_count() == 1);

  const auto expected_scalar =
      tf::cpp::ray_cast(ray<TestType>(), mesh).scalar();
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::ray_cast(counting_resolver{submissions},
                                         ray<TestType>(), mesh);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  const auto custom_scalar = custom.get().scalar();
  CHECK(custom_scalar.hit == expected_scalar.hit);
  CHECK(custom_scalar.t == Catch::Approx(expected_scalar.t));
  CHECK(custom_scalar.element_id == expected_scalar.element_id);
  CHECK(mesh_storage.cache.tree_build_count() == 1);

  // the carrier is borrowed, so the storage it names is kept alive by the
  // keepalive it was assembled with and no handle of the caller's survives
  auto held = tf::cpp::async::ray_cast(
      ray<TestType>(), held_carrier_of(triangle_mesh<TestType>()));
  CHECK(held.get().scalar().element_id == 0);

  const auto nothing_storage =
      tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType>{};
  CHECK(tf::cpp::async::ray_cast(ray<TestType>(), nothing_storage.mesh())
            .get()
            .scalar()
            .element_id == -1);
}

TEMPLATE_TEST_CASE("async near-epsilon inverted ray bounds always miss",
                   "[cpp][spatial][ray_cast][async][bounds][inverted]", float,
                   double) {
  const auto inverted_min =
      std::nextafter(TestType{1}, std::numeric_limits<TestType>::max());
  tf::cpp::ray_cast_options<TestType> scalar_options;
  scalar_options.min_t = inverted_min;
  scalar_options.max_t = TestType{1};
  const auto mesh_storage = triangle_mesh<TestType>();

  auto primitive = tf::cpp::async::ray_cast(
      ray<TestType>(), triangle<TestType>(), scalar_options);
  auto form = tf::cpp::async::ray_cast(ray<TestType>(), mesh_storage.mesh(),
                                       scalar_options);
  const auto primitive_value = primitive.get().scalar();
  const auto form_value = form.get().scalar();
  CHECK_FALSE(primitive_value.hit);
  CHECK(primitive_value.t == Catch::Approx(1));
  CHECK_FALSE(form_value.hit);
  CHECK(form_value.t == TestType{0});

  tf::cpp::ray_cast_options<TestType> batch_options;
  batch_options.min_ts =
      make_array<TestType>({inverted_min, inverted_min}, {2});
  batch_options.max_ts = make_array<TestType>({TestType{1}, TestType{1}}, {2});
  const auto batch_mesh_storage = triangle_mesh<TestType>();
  auto primitive_batch = tf::cpp::async::ray_cast(
      rays<TestType>(), triangle<TestType>(), batch_options);
  auto form_batch = tf::cpp::async::ray_cast(
      rays<TestType>(), batch_mesh_storage.mesh(), batch_options);
  const auto primitive_batch_value = primitive_batch.get().batch();
  const auto form_batch_value = form_batch.get().batch();
  CHECK(primitive_batch_value.hits[0] == 0);
  CHECK(primitive_batch_value.hits[1] == 0);
  CHECK(primitive_batch_value.ts[0] == Catch::Approx(1));
  CHECK(primitive_batch_value.ts[1] == Catch::Approx(2));
  CHECK(form_batch_value.hits[0] == 0);
  CHECK(form_batch_value.hits[1] == 0);
  CHECK(form_batch_value.ts[0] == TestType{0});
  CHECK(form_batch_value.ts[1] == TestType{0});
}

// N carriers over ONE geometry and ONE cache, each with its own frame: a cache
// is built in local coordinates, so the instances share the tree.
TEMPLATE_TEST_CASE("instances of one geometry share its filled cache",
                   "[cpp][spatial][ray_cast][concurrency]", float, double) {
  const auto storage = triangle_mesh<TestType>();
  const auto shifted = translation<TestType>(2);
  const auto at_origin = storage.mesh();
  const auto at_two = tf::cpp::mesh<tf::cpp::default_index_t, TestType>{
      storage.polygons.faces(), storage.polygons.points(), storage.cache,
      tf::make_transformation_view<3>(shifted.data())};
  // both instances are warmed before either is shared, so the threads only
  // read filled state and neither asks the cache
  tf::cpp::build_tree(at_origin);
  tf::cpp::build_tree(at_two);
  REQUIRE(storage.cache.tree_build_count() == 1);

  std::atomic<bool> stop{false};
  std::atomic<int> reads{0};
  std::thread reader([&at_two, &stop, &reads] {
    while (!stop.load(std::memory_order_relaxed)) {
      static_cast<void>(tf::cpp::ray_cast(ray<TestType>(), at_two));
      reads.fetch_add(1, std::memory_order_relaxed);
    }
  });
  while (reads.load(std::memory_order_relaxed) == 0)
    std::this_thread::yield();

  for (int iteration = 0; iteration < 64; ++iteration) {
    const auto result = tf::cpp::ray_cast(ray<TestType>(), at_origin).scalar();
    CHECK(result.hit);
    CHECK(result.t == Catch::Approx(1));
  }
  stop.store(true, std::memory_order_relaxed);
  reader.join();
  CHECK(reads.load(std::memory_order_relaxed) > 0);
  CHECK(tf::cpp::ray_cast(ray<TestType>(), at_two).scalar().t ==
        Catch::Approx(3));
  CHECK(storage.cache.tree_build_count() == 1);
}

namespace {

template <typename Index, typename Real, std::size_t Dims>
struct ray_cast_form_axis {
  using real_type = Real;
  using index_type = Index;
  static constexpr std::size_t dims = Dims;
};

using ray_cast_form_axes =
    std::tuple<ray_cast_form_axis<std::int32_t, float, 2>,
               ray_cast_form_axis<std::int64_t, float, 2>,
               ray_cast_form_axis<std::int32_t, double, 2>,
               ray_cast_form_axis<std::int64_t, double, 2>,
               ray_cast_form_axis<std::int32_t, float, 3>,
               ray_cast_form_axis<std::int64_t, float, 3>,
               ray_cast_form_axis<std::int32_t, double, 3>,
               ray_cast_form_axis<std::int64_t, double, 3>>;

template <typename T>
auto make_vector_array(const std::vector<T> &values,
                       tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  std::copy(values.begin(), values.end(), buffer.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real, std::size_t Dims>
auto generalized_ray(Real distance = Real{1})
    -> tf::cpp::primitive<Real, Dims> {
  if constexpr (Dims == 2)
    return tf::cpp::primitive<Real, Dims>(
        tf::cpp::primitive_kind::ray,
        make_array<Real>({-distance, 0, 1, 0}, {2, 2}));
  else
    return tf::cpp::primitive<Real, Dims>(
        tf::cpp::primitive_kind::ray,
        make_array<Real>({0, 0, -distance, 0, 0, 1}, {2, 3}));
}

template <typename Real, std::size_t Dims>
auto generalized_rays() -> tf::cpp::primitive<Real, Dims> {
  if constexpr (Dims == 2)
    return tf::cpp::primitive<Real, Dims>(
        tf::cpp::primitive_kind::ray,
        make_array<Real>({-1, 0, 1, 0, -2, 0, 1, 0}, {2, 2, 2}));
  else
    return tf::cpp::primitive<Real, Dims>(
        tf::cpp::primitive_kind::ray,
        make_array<Real>({0, 0, -1, 0, 0, 1, 0, 0, -2, 0, 0, 1}, {2, 2, 3}));
}

template <typename Index, typename Real, std::size_t Dims>
auto generalized_fixed_mesh() -> tf::cpp::test::owned_mesh<Index, Real, Dims> {
  if constexpr (Dims == 2)
    return {tf::cpp::test::polygons_of<Index, Real, Dims>({0, 1, 2},
                                                          {0, -1, 1, 0, 0, 1})};
  else
    return {tf::cpp::test::polygons_of<Index, Real, Dims>(
        {0, 1, 2}, {-1, -1, 0, 1, -1, 0, 0, 1, 0})};
}

template <typename Index, typename Real, std::size_t Dims>
auto generalized_dynamic_mesh()
    -> tf::cpp::test::owned_mesh<Index, Real, Dims, tf::dynamic_size> {
  if constexpr (Dims == 2)
    return {tf::cpp::test::polygons_of<Index, Real, Dims>(
        {0, 4}, {0, 1, 2, 3}, {0, -1, 1, -1, 1, 1, 0, 1})};
  else
    return {tf::cpp::test::polygons_of<Index, Real, Dims>(
        {0, 4}, {0, 1, 2, 3}, {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0})};
}

template <typename Index, typename Real, std::size_t Dims>
auto generalized_edge_mesh()
    -> tf::cpp::test::owned_edge_mesh<Index, Real, Dims> {
  if constexpr (Dims == 2)
    return {
        tf::cpp::test::segments_of<Index, Real, Dims>({0, 1}, {0, -1, 0, 1})};
  else
    return {tf::cpp::test::segments_of<Index, Real, Dims>({0, 1},
                                                          {-1, 0, 0, 1, 0, 0})};
}

template <typename Real, std::size_t Dims>
auto generalized_cloud() -> tf::cpp::test::owned_point_cloud<Real, Dims> {
  if constexpr (Dims == 2)
    return {tf::cpp::test::points_of<Real, Dims>({0, 0})};
  else
    return {tf::cpp::test::points_of<Real, Dims>({0, 0, 0})};
}

/// The placement a carrier is assembled at, in the shape `place` states it.
template <typename Real, std::size_t Dims>
auto generalized_translation(Real distance)
    -> std::array<Real, (Dims + 1) * (Dims + 1)> {
  constexpr auto side = Dims + 1;
  std::array<Real, side * side> values{};
  for (std::size_t diagonal = 0; diagonal < side; ++diagonal)
    values[diagonal * side + diagonal] = Real{1};
  const auto axis = Dims == 2 ? std::size_t{0} : std::size_t{2};
  values[axis * side + Dims] = distance;
  return values;
}

/// The same, for an edge carrier whose storage must outlive the caller.
template <typename Index, typename Real, std::size_t Dims>
auto held_carrier_of(tf::cpp::test::owned_edge_mesh<Index, Real, Dims> owner)
    -> tf::cpp::edge_mesh<Index, Real, Dims> {
  const auto held =
      std::make_shared<const tf::cpp::test::owned_edge_mesh<Index, Real, Dims>>(
          std::move(owner));
  return {held->segments.edges(), held->segments.points(), held->cache,
          held->frame(), held};
}

template <typename Real, std::size_t Dims>
auto held_carrier_of(tf::cpp::test::owned_point_cloud<Real, Dims> owner)
    -> tf::cpp::point_cloud<Real, Dims> {
  const auto held =
      std::make_shared<const tf::cpp::test::owned_point_cloud<Real, Dims>>(
          std::move(owner));
  return {held->points.points(), held->normals.unit_vectors(), held->cache,
          held->frame(), held};
}

struct mutating_ray_cast_resolver {
  std::shared_ptr<std::atomic<int>> submissions;
  std::function<void()> mutate;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    mutate();
    return std::make_shared<state_type<T>>();
  }
};

} // namespace

TEMPLATE_LIST_TEST_CASE(
    "generalized form ray cast covers Real Index Dims and all carriers",
    "[cpp][spatial][ray_cast][form][matrix]", ray_cast_form_axes) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  using QueryReal =
      std::conditional_t<std::is_same_v<Real, float>, double, float>;
  using result_type = tf::cpp::ray_cast_form_result<Index, Real>;

  auto fixed_storage = generalized_fixed_mesh<Index, Real, Dims>();
  auto dynamic_storage = generalized_dynamic_mesh<Index, Real, Dims>();
  auto edge_storage = generalized_edge_mesh<Index, Real, Dims>();
  auto cloud_storage = generalized_cloud<Real, Dims>();
  const auto fixed = fixed_storage.mesh();
  const auto dynamic = dynamic_storage.mesh();
  const auto edges = edge_storage.edge_mesh();
  const auto cloud = cloud_storage.point_cloud();
  const auto query = generalized_ray<QueryReal, Dims>();

  const auto deduced_fixed_result = tf::cpp::ray_cast(query, fixed);
  const auto fixed_result =
      tf::cpp::ray_cast<QueryReal, Index, Real, Dims>(query, fixed);
  const auto dynamic_result =
      tf::cpp::ray_cast<QueryReal, Index, Real, Dims>(query, dynamic);
  const auto edge_result =
      tf::cpp::ray_cast<QueryReal, Index, Real, Dims>(query, edges);
  const auto cloud_result =
      tf::cpp::ray_cast<QueryReal, Real, Dims>(query, cloud);
  STATIC_REQUIRE(
      std::is_same_v<decltype(deduced_fixed_result), const result_type>);
  STATIC_REQUIRE(std::is_same_v<decltype(fixed_result), const result_type>);
  REQUIRE(deduced_fixed_result.is_scalar());
  REQUIRE(fixed_result.is_scalar());
  REQUIRE(dynamic_result.is_scalar());
  REQUIRE(edge_result.is_scalar());
  REQUIRE(cloud_result.is_scalar());
  STATIC_REQUIRE(
      std::is_same_v<std::remove_cv_t<std::remove_reference_t<
                         decltype(fixed_result.scalar().element_id)>>,
                     Index>);
  CHECK(deduced_fixed_result.scalar().hit);
  CHECK(fixed_result.scalar().hit);
  CHECK(dynamic_result.scalar().hit);
  CHECK(edge_result.scalar().hit);
  CHECK(cloud_result.scalar().hit);
  CHECK(fixed_result.scalar().element_id == Index{0});
  CHECK(dynamic_result.scalar().element_id == Index{0});
  CHECK(edge_result.scalar().element_id == Index{0});
  CHECK(cloud_result.scalar().element_id == std::int32_t{0});
  CHECK(fixed_result.scalar().t == Catch::Approx(1));
  CHECK(dynamic_result.scalar().t == Catch::Approx(1));
  CHECK(edge_result.scalar().t == Catch::Approx(1));
  CHECK(cloud_result.scalar().t == Catch::Approx(1));

  const auto batch_query = generalized_rays<QueryReal, Dims>();
  const auto mesh_batch =
      tf::cpp::ray_cast<QueryReal, Index, Real, Dims>(batch_query, fixed);
  const auto edge_batch =
      tf::cpp::ray_cast<QueryReal, Index, Real, Dims>(batch_query, edges);
  const auto cloud_batch =
      tf::cpp::ray_cast<QueryReal, Real, Dims>(batch_query, cloud);
  REQUIRE(mesh_batch.is_batch());
  REQUIRE(edge_batch.is_batch());
  REQUIRE(cloud_batch.is_batch());
  STATIC_REQUIRE(
      std::is_same_v<std::remove_cv_t<std::remove_reference_t<
                         decltype(mesh_batch.batch().element_ids[0])>>,
                     Index>);
  CHECK(mesh_batch.batch().element_ids[0] == Index{0});
  CHECK(mesh_batch.batch().element_ids[1] == Index{0});
  CHECK(edge_batch.batch().element_ids[0] == Index{0});
  CHECK(cloud_batch.batch().element_ids[0] == std::int32_t{0});
  CHECK(mesh_batch.batch().ts[0] == Catch::Approx(1));
  CHECK(mesh_batch.batch().ts[1] == Catch::Approx(2));

  // the placement is this instance's, so a moved carrier is a new assembly
  const auto transform = generalized_translation<Real, Dims>(Real{2});
  fixed_storage.place(transform);
  dynamic_storage.place(transform);
  edge_storage.place(transform);
  cloud_storage.place(transform);
  CHECK(tf::cpp::ray_cast<QueryReal, Index, Real, Dims>(query,
                                                        fixed_storage.mesh())
            .scalar()
            .t == Catch::Approx(3));
  CHECK(tf::cpp::ray_cast<QueryReal, Index, Real, Dims>(query,
                                                        dynamic_storage.mesh())
            .scalar()
            .t == Catch::Approx(3));
  CHECK(tf::cpp::ray_cast<QueryReal, Index, Real, Dims>(
            query, edge_storage.edge_mesh())
            .scalar()
            .t == Catch::Approx(3));
  CHECK(tf::cpp::ray_cast<QueryReal, Real, Dims>(query,
                                                 cloud_storage.point_cloud())
            .scalar()
            .t == Catch::Approx(3));
  CHECK(tf::cpp::ray_cast<QueryReal, Index, Real, Dims>(query, fixed)
            .scalar()
            .t == Catch::Approx(1));
}

TEMPLATE_LIST_TEST_CASE(
    "generalized edge mesh ray cast preserves interpolation endpoints and IDs",
    "[cpp][spatial][ray_cast][edge-mesh][python-parity]", ray_cast_form_axes) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  const auto edge_storage = [&] {
    if constexpr (Dims == 2)
      return tf::cpp::test::owned_edge_mesh<Index, Real, Dims>{
          tf::cpp::test::segments_of<Index, Real, Dims>(
              {0, 1, 2, 3}, {5, -1, 5, 1, 0, -1, 0, 1})};
    else
      return tf::cpp::test::owned_edge_mesh<Index, Real, Dims>{
          tf::cpp::test::segments_of<Index, Real, Dims>(
              {0, 1, 2, 3}, {5, -1, 0, 5, 1, 0, -1, 0, 0, 1, 0, 0})};
  }();
  const auto edges = edge_storage.edge_mesh();

  auto midpoint = generalized_ray<Real, Dims>();
  const auto midpoint_result =
      tf::cpp::ray_cast<Real, Index, Real, Dims>(midpoint, edges).scalar();
  REQUIRE(midpoint_result.hit);
  CHECK(midpoint_result.element_id == Index{1});
  CHECK(midpoint_result.t == Catch::Approx(1));
  if constexpr (Dims == 2) {
    CHECK(midpoint.data()[0] + midpoint_result.t * midpoint.data()[2] ==
          Catch::Approx(0));
    CHECK(midpoint.data()[1] + midpoint_result.t * midpoint.data()[3] ==
          Catch::Approx(0));
  } else {
    CHECK(midpoint.data()[0] + midpoint_result.t * midpoint.data()[3] ==
          Catch::Approx(0));
    CHECK(midpoint.data()[2] + midpoint_result.t * midpoint.data()[5] ==
          Catch::Approx(0));
  }

  auto endpoint = [&] {
    if constexpr (Dims == 2)
      return tf::cpp::primitive<Real, Dims>(
          tf::cpp::primitive_kind::ray,
          make_array<Real>({-1, -1, 1, 0}, {2, 2}));
    else
      return tf::cpp::primitive<Real, Dims>(
          tf::cpp::primitive_kind::ray,
          make_array<Real>({-1, 0, -2, 0, 0, 1}, {2, 3}));
  }();
  const auto endpoint_result =
      tf::cpp::ray_cast<Real, Index, Real, Dims>(endpoint, edges).scalar();
  REQUIRE(endpoint_result.hit);
  CHECK(endpoint_result.element_id == Index{1});
  CHECK(endpoint_result.t == Catch::Approx(Dims == 2 ? 1 : 2));
}

TEST_CASE("generalized form ray cast selects nearest elements and validates",
          "[cpp][spatial][ray_cast][form][validation][python-parity]") {
  using Real = double;
  using Index = std::int64_t;
  constexpr std::size_t Dims = 3;

  const auto mesh_storage = tf::cpp::test::owned_mesh<Index, Real, Dims>{
      tf::cpp::test::polygons_of<Index, Real, Dims>(
          {0, 1, 2, 3, 4, 5},
          {-1, -1, 2, 1, -1, 2, 0, 1, 2, -1, -1, 0, 1, -1, 0, 0, 1, 0})};
  const auto cloud_storage = tf::cpp::test::owned_point_cloud<Real, Dims>{
      tf::cpp::test::points_of<Real, Dims>({0, 0, 2, 0, 0, 0})};
  const auto mesh = mesh_storage.mesh();
  const auto cloud = cloud_storage.point_cloud();
  const auto query = generalized_ray<float, Dims>();
  CHECK(tf::cpp::ray_cast<float, Index, Real, Dims>(query, mesh)
            .scalar()
            .element_id == Index{1});
  CHECK(
      tf::cpp::ray_cast<float, Real, Dims>(query, cloud).scalar().element_id ==
      std::int32_t{1});

  tf::cpp::ray_cast_options<Real> inclusive;
  inclusive.min_t = Real{1};
  inclusive.max_t = Real{1};
  CHECK(tf::cpp::ray_cast<float, Index, Real, Dims>(query, mesh, inclusive)
            .scalar()
            .hit);
  tf::cpp::ray_cast_options<Real> inverted;
  inverted.min_t = std::nextafter(Real{1}, Real{2});
  inverted.max_t = Real{1};
  const auto miss =
      tf::cpp::ray_cast<float, Index, Real, Dims>(query, mesh, inverted)
          .scalar();
  CHECK_FALSE(miss.hit);
  CHECK(miss.element_id == Index{-1});

  tf::cpp::ray_cast_options<Real> wrong_bounds;
  wrong_bounds.max_ts = make_array<Real>({1}, {1});
  CHECK_THROWS_AS((tf::cpp::ray_cast<float, Index, Real, Dims>(
                      generalized_rays<float, Dims>(), mesh, wrong_bounds)),
                  std::invalid_argument);

  // a cast at an empty carrier simply misses
  const auto empty_mesh_storage =
      tf::cpp::test::owned_mesh<Index, Real, Dims>{};
  const auto empty_edge_storage =
      tf::cpp::test::owned_edge_mesh<Index, Real, Dims>{};
  const auto empty_cloud_storage =
      tf::cpp::test::owned_point_cloud<Real, Dims>{};
  CHECK_FALSE((tf::cpp::ray_cast<float, Index, Real, Dims>(
                   query, empty_mesh_storage.mesh()))
                  .scalar()
                  .hit);
  CHECK_FALSE((tf::cpp::ray_cast<float, Index, Real, Dims>(
                   query, empty_edge_storage.edge_mesh()))
                  .scalar()
                  .hit);
  CHECK_FALSE((tf::cpp::ray_cast<float, Real, Dims>(
                   query, empty_cloud_storage.point_cloud()))
                  .scalar()
                  .hit);

  // a face naming a point the mesh does not have is refused per reading,
  // before any tree kernel runs
  const auto out_of_reach_storage =
      tf::cpp::test::owned_mesh<Index, Real, Dims, tf::dynamic_size>{
          tf::cpp::test::polygons_of<Index, Real, Dims>(
              std::initializer_list<Index>{0, 3},
              std::initializer_list<Index>{0, 1, 4},
              std::initializer_list<Real>{0, 0, 0, 1, 0, 0, 0, 1, 0})};
  CHECK_THROWS_AS((tf::cpp::ray_cast<float, Index, Real, Dims>(
                      query, out_of_reach_storage.mesh())),
                  std::out_of_range);
  CHECK(out_of_reach_storage.cache.tree_build_count() == 0);
}

TEST_CASE("generalized async form ray casts answer from the reading they were "
          "handed",
          "[cpp][spatial][ray_cast][form][async]") {
  using Real = double;
  using Index = std::int64_t;
  constexpr std::size_t Dims = 2;
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  const auto query = generalized_ray<float, Dims>();

  // the carriers are borrowed, so the storage they name is kept alive by the
  // keepalive they were assembled with and no handle of the caller's survives
  auto mesh_pending = tf::cpp::async::ray_cast(
      counting_resolver{submissions}, query,
      held_carrier_of(generalized_dynamic_mesh<Index, Real, Dims>()));
  CHECK(mesh_pending.get().scalar().t == Catch::Approx(1));

  auto edge_pending = tf::cpp::async::ray_cast(
      counting_resolver{submissions}, query,
      held_carrier_of(generalized_edge_mesh<Index, Real, Dims>()));
  CHECK(edge_pending.get().scalar().t == Catch::Approx(1));

  auto cloud_pending = tf::cpp::async::ray_cast(
      counting_resolver{submissions}, query,
      held_carrier_of(generalized_cloud<Real, Dims>()));
  CHECK(cloud_pending.get().scalar().t == Catch::Approx(1));
  CHECK(submissions->load(std::memory_order_relaxed) == 3);
}

TEST_CASE("generalized async form ray cast retains its ray and bound arrays",
          "[cpp][spatial][ray_cast][form][async][ownership]") {
  using Real = double;
  using Index = std::int64_t;
  constexpr std::size_t Dims = 2;
  const auto submissions = std::make_shared<std::atomic<int>>(0);

  // THE ASYNC ARRAY LAW: a job carries the handles, so the caller may release
  // its own before the job runs and the storage the job reads stays
  auto query = generalized_ray<float, Dims>();
  const auto mesh_storage = generalized_fixed_mesh<Index, Real, Dims>();
  const auto mesh = mesh_storage.mesh();
  auto held_query = query;
  auto ray_pending = tf::cpp::async::ray_cast(
      mutating_ray_cast_resolver{submissions,
                                 [&] {
                                   query = generalized_ray<float, Dims>();
                                   query.data().destroy();
                                 }},
      held_query, mesh);
  CHECK(ray_pending.get().scalar().t == Catch::Approx(1));

  auto batch_query = generalized_rays<float, Dims>();
  const auto edge_storage = generalized_edge_mesh<Index, Real, Dims>();
  const auto edges = edge_storage.edge_mesh();
  tf::cpp::ray_cast_options<Real> options;
  options.min_ts = make_array<Real>({Real{0}, Real{0}}, {2});
  options.max_ts = make_array<Real>({Real{1}, Real{2}}, {2});
  auto held_options = options;
  auto bounds_pending = tf::cpp::async::ray_cast(
      mutating_ray_cast_resolver{submissions,
                                 [&] {
                                   options.min_ts.destroy();
                                   options.max_ts.destroy();
                                 }},
      batch_query, edges, held_options);
  const auto bounds_result = bounds_pending.get().batch();
  CHECK(bounds_result.hits[0] == 1);
  CHECK(bounds_result.hits[1] == 1);
  CHECK(bounds_result.ts[0] == Catch::Approx(1));
  CHECK(bounds_result.ts[1] == Catch::Approx(2));
  CHECK(submissions->load(std::memory_order_relaxed) == 2);
}
