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
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/spatial/async/intersects.hpp"
#include "trueform/cpp/spatial/intersects.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <future>
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

template <typename T>
auto make_empty(tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(0);
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real>
auto origin_primitive(tf::cpp::primitive_kind kind)
    -> tf::cpp::primitive<Real> {
  using tf::cpp::primitive;
  switch (kind) {
  case tf::cpp::primitive_kind::point:
    return primitive<Real>(kind, make_array<Real>({0, 0, 0}, {3}));
  case tf::cpp::primitive_kind::vector:
    return primitive<Real>(kind, make_array<Real>({1, 0, 0}, {3}));
  case tf::cpp::primitive_kind::segment:
    return primitive<Real>(kind, make_array<Real>({-1, 0, 0, 1, 0, 0}, {2, 3}));
  case tf::cpp::primitive_kind::triangle:
    return primitive<Real>(
        kind, make_array<Real>({-1, -1, 0, 1, -1, 0, 0, 1, 0}, {3, 3}));
  case tf::cpp::primitive_kind::ray:
  case tf::cpp::primitive_kind::line:
    return primitive<Real>(kind, make_array<Real>({0, 0, 0, 1, 0, 0}, {2, 3}));
  case tf::cpp::primitive_kind::plane:
    return primitive<Real>(kind, make_array<Real>({0, 0, 1, 0}, {4}));
  case tf::cpp::primitive_kind::aabb:
    return primitive<Real>(
        kind, make_array<Real>({-0.5, -0.5, -0.5, 0.5, 0.5, 0.5}, {2, 3}));
  case tf::cpp::primitive_kind::polygon:
    return primitive<Real>(
        kind,
        make_array<Real>({-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0}, {4, 3}));
  }
  throw std::logic_error("unknown primitive kind");
}

template <typename Real>
auto triangle_mesh()
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {-1, -1, 0, 1, -1, 0, 0, 1, 0})};
}

/// What a caller assembles when no handle of its own outlives the call: the
/// storage lives in a shared block and the carrier retains it.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto held_carrier_of(tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon> owner)
    -> tf::cpp::mesh<Index, Real, Dims, Ngon> {
  const auto held = std::make_shared<
      const tf::cpp::test::owned_mesh<Index, Real, Dims, Ngon>>(
      std::move(owner));
  return {held->polygons.faces(), held->polygons.points(), held->cache,
          held->frame(), held};
}

template <typename Index, typename Real, std::size_t Dims>
auto held_carrier_of(tf::cpp::test::owned_edge_mesh<Index, Real, Dims> owner)
    -> tf::cpp::edge_mesh<Index, Real, Dims> {
  const auto held = std::make_shared<
      const tf::cpp::test::owned_edge_mesh<Index, Real, Dims>>(
      std::move(owner));
  return {held->segments.edges(), held->segments.points(), held->cache,
          held->frame(), held};
}

template <typename Real>
auto origin_cloud() -> tf::cpp::test::owned_point_cloud<Real> {
  return {tf::cpp::test::points_of<Real>({0, 0, 0})};
}

constexpr std::array<tf::cpp::primitive_kind, 8> spatial_kinds{
    tf::cpp::primitive_kind::point,    tf::cpp::primitive_kind::segment,
    tf::cpp::primitive_kind::triangle, tf::cpp::primitive_kind::ray,
    tf::cpp::primitive_kind::line,     tf::cpp::primitive_kind::plane,
    tf::cpp::primitive_kind::aabb,     tf::cpp::primitive_kind::polygon};

constexpr std::array<tf::cpp::primitive_kind, 7> spatial_kinds_2d{
    tf::cpp::primitive_kind::point,    tf::cpp::primitive_kind::segment,
    tf::cpp::primitive_kind::triangle, tf::cpp::primitive_kind::ray,
    tf::cpp::primitive_kind::line,     tf::cpp::primitive_kind::aabb,
    tf::cpp::primitive_kind::polygon};

template <typename Real>
auto origin_primitive_2d(tf::cpp::primitive_kind kind)
    -> tf::cpp::primitive<Real, 2> {
  using primitive_type = tf::cpp::primitive<Real, 2>;
  switch (kind) {
  case tf::cpp::primitive_kind::point:
    return primitive_type(kind, make_array<Real>({0, 0}, {2}));
  case tf::cpp::primitive_kind::segment:
    return primitive_type(kind, make_array<Real>({-1, 0, 1, 0}, {2, 2}));
  case tf::cpp::primitive_kind::triangle:
    return primitive_type(kind,
                          make_array<Real>({-1, -1, 1, -1, 0, 1}, {3, 2}));
  case tf::cpp::primitive_kind::ray:
  case tf::cpp::primitive_kind::line:
    return primitive_type(kind, make_array<Real>({0, 0, 1, 0}, {2, 2}));
  case tf::cpp::primitive_kind::aabb:
    return primitive_type(kind,
                          make_array<Real>({-0.5, -0.5, 0.5, 0.5}, {2, 2}));
  case tf::cpp::primitive_kind::polygon:
    return primitive_type(
        kind, make_array<Real>({-1, -1, 1, -1, 1, 1, -1, 1}, {4, 2}));
  case tf::cpp::primitive_kind::vector:
  case tf::cpp::primitive_kind::plane:
    break;
  }
  throw std::logic_error("unsupported 2D primitive kind");
}

template <typename A, typename B>
auto check_intersection_pair(const A &a, const B &b, bool expected) -> void {
  CHECK(tf::cpp::intersects(a, b).scalar() == expected);
  CHECK(tf::cpp::intersects(b, a).scalar() == expected);
}

template <typename A, typename B, typename = void>
struct has_intersects : std::false_type {};

template <typename A, typename B>
struct has_intersects<
    A, B,
    std::void_t<decltype(tf::cpp::intersects(std::declval<const A &>(),
                                             std::declval<const B &>()))>>
    : std::true_type {};

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

} // namespace

TEST_CASE("runtime primitives validate canonical 3D layouts and ownership",
          "[cpp][spatial][primitive]") {
  auto data = make_array<float>({0, 0, 0, 1, 1, 1}, {2, 3});
  tf::cpp::primitive<float> points(tf::cpp::primitive_kind::point, data);
  CHECK(points.kind() == tf::cpp::primitive_kind::point);
  CHECK(points.cardinality() == tf::cpp::primitive_cardinality::batch);
  CHECK(points.is_batch());
  CHECK(points.count() == 2);
  CHECK(points.polygon_vertex_count() == 0);

  auto element = points.at(0);
  auto batch_one = points.slice(0, 1);
  CHECK_FALSE(element.is_batch());
  CHECK(batch_one.is_batch());
  CHECK(batch_one.count() == 1);

  auto shallow = points.shallow_copy();
  auto deep = points.deep_copy();
  shallow.data()[0] = 4;
  deep.data()[1] = 7;
  CHECK(points.data()[0] == 4);
  CHECK(points.data()[1] == 0);

  auto polygon = origin_primitive<float>(tf::cpp::primitive_kind::polygon);
  CHECK(polygon.polygon_vertex_count() == 4);

  CHECK_NOTHROW((tf::cpp::primitive<float>(tf::cpp::primitive_kind::vector,
                                           make_array<float>({1, 2, 3}, {3}))));
  CHECK_THROWS_AS((tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                             make_array<float>({0, 0}, {2}))),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::primitive<float>(tf::cpp::primitive_kind::segment,
                                 make_array<float>({0, 0, 1, 1}, {2, 2}))),
      std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::primitive<float>(
                      tf::cpp::primitive_kind::triangle,
                      make_array<float>({0, 0, 1, 0, 0, 1}, {3, 2}))),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::primitive<float>(tf::cpp::primitive_kind::plane,
                                 make_array<float>({0, 0, 1, 0}, {2, 2}))),
      std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::primitive<float>(
                      tf::cpp::primitive_kind::polygon,
                      make_array<float>({0, 0, 0, 1, 0, 0}, {2, 3}))),
                  std::invalid_argument);
}

TEST_CASE("primitive intersection dispatch covers every ordered spatial pair",
          "[cpp][spatial][intersects][dispatch]") {
  for (const auto kind_a : spatial_kinds) {
    for (const auto kind_b : spatial_kinds) {
      INFO("ordered primitive pair " << static_cast<int>(kind_a) << " x "
                                     << static_cast<int>(kind_b));
      const auto result = tf::cpp::intersects(origin_primitive<double>(kind_a),
                                              origin_primitive<double>(kind_b));
      CHECK(result.is_scalar());
    }
  }

  auto hit_a = origin_primitive<float>(tf::cpp::primitive_kind::aabb);
  auto hit_b = origin_primitive<float>(tf::cpp::primitive_kind::point);
  auto miss = tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                        make_array<float>({5, 5, 5}, {3}));
  CHECK(tf::cpp::intersects(hit_a, hit_b).scalar());
  CHECK_FALSE(tf::cpp::intersects(hit_a, miss).scalar());
}

TEMPLATE_TEST_CASE(
    "2D primitive intersections match Python hit miss touch and batch cases",
    "[cpp][spatial][intersects][2d][python-parity]", float, double) {
  for (const auto kind_a : spatial_kinds_2d) {
    for (const auto kind_b : spatial_kinds_2d) {
      INFO("ordered 2D primitive pair " << static_cast<int>(kind_a) << " x "
                                        << static_cast<int>(kind_b));
      const auto result =
          tf::cpp::intersects(origin_primitive_2d<TestType>(kind_a),
                              origin_primitive_2d<TestType>(kind_b));
      CHECK(result.is_scalar());
    }
  }

  const auto point = [&](std::initializer_list<TestType> values) {
    return tf::cpp::primitive<TestType, 2>(tf::cpp::primitive_kind::point,
                                           make_array<TestType>(values, {2}));
  };
  const auto pair = [&](tf::cpp::primitive_kind kind,
                        std::initializer_list<TestType> values) {
    return tf::cpp::primitive<TestType, 2>(
        kind, make_array<TestType>(values, {2, 2}));
  };

  check_intersection_pair(point({1, 2}), point({1, 2}), true);
  check_intersection_pair(point({1, 2}), point({5, 6}), false);

  const auto diagonal = pair(tf::cpp::primitive_kind::segment, {0, 0, 1, 1});
  check_intersection_pair(point({TestType{0.5}, TestType{0.5}}), diagonal,
                          true);
  check_intersection_pair(point({TestType{0.5}, 0}), diagonal, false);
  const auto crossing = pair(tf::cpp::primitive_kind::segment, {0, 1, 1, 0});
  const auto disjoint = pair(tf::cpp::primitive_kind::segment, {2, 2, 3, 3});
  check_intersection_pair(diagonal, crossing, true);
  check_intersection_pair(diagonal, disjoint, false);

  const auto square = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::polygon,
      make_array<TestType>({0, 0, 1, 0, 1, 1, 0, 1}, {4, 2}));
  check_intersection_pair(point({TestType{0.5}, TestType{0.5}}), square, true);
  check_intersection_pair(point({2, 2}), square, false);

  const auto vertical_segment = pair(tf::cpp::primitive_kind::segment,
                                     {TestType{0.5}, 0, TestType{0.5}, 1});
  const auto ray_hit =
      pair(tf::cpp::primitive_kind::ray, {0, TestType{0.5}, 1, 0});
  const auto ray_miss = pair(tf::cpp::primitive_kind::ray, {0, 2, 1, 0});
  check_intersection_pair(ray_hit, vertical_segment, true);
  check_intersection_pair(ray_miss, vertical_segment, false);

  const auto line_horizontal =
      pair(tf::cpp::primitive_kind::line, {0, 0, 1, 0});
  const auto line_vertical =
      pair(tf::cpp::primitive_kind::line, {TestType{0.5}, -1, 0, 1});
  const auto line_parallel = pair(tf::cpp::primitive_kind::line, {0, 1, 1, 0});
  check_intersection_pair(line_horizontal, line_vertical, true);
  check_intersection_pair(line_horizontal, line_parallel, false);

  auto box = origin_primitive_2d<TestType>(tf::cpp::primitive_kind::aabb);
  auto boundary = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::point,
      make_array<TestType>({TestType{0.5}, 0}, {2}));
  auto outside = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::point, make_array<TestType>({5, 5}, {2}));
  CHECK(tf::cpp::intersects(box, boundary).scalar());
  CHECK(tf::cpp::intersects(boundary, box).scalar());
  CHECK_FALSE(tf::cpp::intersects(box, outside).scalar());

  const auto unit_box = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::aabb,
      make_array<TestType>({0, 0, 1, 1}, {2, 2}));
  const auto segment_box_hit =
      pair(tf::cpp::primitive_kind::segment,
           {TestType{0.5}, TestType{-0.5}, TestType{0.5}, TestType{1.5}});
  const auto segment_box_miss = pair(tf::cpp::primitive_kind::segment,
                                     {2, TestType{0.5}, 3, TestType{0.5}});
  check_intersection_pair(segment_box_hit, unit_box, true);
  check_intersection_pair(segment_box_miss, unit_box, false);
  const auto line_box_hit =
      pair(tf::cpp::primitive_kind::line, {TestType{0.5}, -1, 0, 1});
  const auto line_box_miss = pair(tf::cpp::primitive_kind::line, {2, 0, 0, 1});
  check_intersection_pair(line_box_hit, unit_box, true);
  check_intersection_pair(line_box_miss, unit_box, false);

  auto points = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::point,
      make_array<TestType>({0, 0, 5, 5, TestType{0.5}, 0}, {3, 2}));
  const auto broadcast = tf::cpp::intersects(points, box);
  REQUIRE(broadcast.is_batch());
  REQUIRE(broadcast.batch().raw_shape() == tf::small_vector<int, 3>{3});
  CHECK(broadcast.batch()[0] == 1);
  CHECK(broadcast.batch()[1] == 0);
  CHECK(broadcast.batch()[2] == 1);

  auto boxes = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::aabb,
      make_array<TestType>({-1, -1, 1, 1, 4, 4, 6, 6, 0, -1, 1, 1}, {3, 2, 2}));
  const auto pairwise = tf::cpp::intersects(points, boxes);
  REQUIRE(pairwise.is_batch());
  CHECK(pairwise.batch()[0] == 1);
  CHECK(pairwise.batch()[1] == 1);
  CHECK(pairwise.batch()[2] == 1);

  auto unequal = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::point,
      make_array<TestType>({0, 0, 1, 1}, {2, 2}));
  CHECK_THROWS_AS(tf::cpp::intersects(points, unequal), std::invalid_argument);

  auto empty = tf::cpp::primitive<TestType, 2>(tf::cpp::primitive_kind::point,
                                               make_empty<TestType>({0, 2}));
  const auto empty_result = tf::cpp::intersects(empty, box);
  REQUIRE(empty_result.is_batch());
  CHECK(empty_result.batch().raw_shape() == tf::small_vector<int, 3>{0});

  const auto async = tf::cpp::async::intersects(points, box).get();
  REQUIRE(async.is_batch());
  REQUIRE(async.batch().length() == broadcast.batch().length());
  for (std::size_t index = 0; index < async.batch().length(); ++index)
    CHECK(async.batch()[index] == broadcast.batch()[index]);
}

TEMPLATE_TEST_CASE(
    "3D primitive intersections match Python plane polygon and AABB fixtures",
    "[cpp][spatial][intersects][3d][python-parity]", float, double) {
  const auto point = [&](std::initializer_list<TestType> values) {
    return tf::cpp::primitive<TestType>(tf::cpp::primitive_kind::point,
                                        make_array<TestType>(values, {3}));
  };
  const auto directed = [&](tf::cpp::primitive_kind kind,
                            std::initializer_list<TestType> values) {
    return tf::cpp::primitive<TestType>(kind,
                                        make_array<TestType>(values, {2, 3}));
  };
  const auto box = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::aabb,
      make_array<TestType>({0, 0, 0, 1, 1, 1}, {2, 3}));

  const auto overlapping_box = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::aabb,
      make_array<TestType>(
          {TestType{0.5}, TestType{0.5}, TestType{0.5}, 2, 2, 2}, {2, 3}));
  const auto disjoint_box = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::aabb,
      make_array<TestType>({5, 5, 5, 6, 6, 6}, {2, 3}));
  check_intersection_pair(box, overlapping_box, true);
  check_intersection_pair(box, disjoint_box, false);

  const auto triangle = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::polygon,
      make_array<TestType>({0, 0, 0, 1, 0, 0, TestType{0.5}, 1, 0}, {3, 3}));
  const auto polygon_ray_hit =
      directed(tf::cpp::primitive_kind::ray,
               {TestType{0.5}, TestType{0.3}, 2, 0, 0, -1});
  const auto polygon_ray_miss =
      directed(tf::cpp::primitive_kind::ray, {5, 5, 2, 0, 0, -1});
  check_intersection_pair(polygon_ray_hit, triangle, true);
  check_intersection_pair(polygon_ray_miss, triangle, false);

  const auto plane = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::plane, make_array<TestType>({0, 0, 1, 0}, {4}));
  check_intersection_pair(plane, point({1, 1, 0}), true);
  check_intersection_pair(plane, point({1, 1, 5}), false);
  const auto plane_ray_hit =
      directed(tf::cpp::primitive_kind::ray, {0, 0, 2, 0, 0, -1});
  const auto plane_ray_parallel =
      directed(tf::cpp::primitive_kind::ray, {0, 0, 2, 1, 0, 0});
  check_intersection_pair(plane, plane_ray_hit, true);
  check_intersection_pair(plane, plane_ray_parallel, false);
  check_intersection_pair(plane, box, true);

  const auto box_ray_hit =
      directed(tf::cpp::primitive_kind::ray,
               {-1, TestType{0.5}, TestType{0.5}, 1, 0, 0});
  const auto box_ray_miss =
      directed(tf::cpp::primitive_kind::ray, {-1, 2, TestType{0.5}, 1, 0, 0});
  check_intersection_pair(box_ray_hit, box, true);
  check_intersection_pair(box_ray_miss, box, false);

  const auto polygon_box_hit = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::polygon,
      make_array<TestType>({TestType{0.5}, TestType{0.5}, TestType{-0.5},
                            TestType{0.5}, TestType{0.5}, TestType{1.5},
                            TestType{1.5}, TestType{0.5}, TestType{0.5}},
                           {3, 3}));
  const auto polygon_box_miss = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::polygon,
      make_array<TestType>({2, 2, 2, 3, 2, 2, TestType{2.5}, 3, 2}, {3, 3}));
  check_intersection_pair(polygon_box_hit, box, true);
  check_intersection_pair(polygon_box_miss, box, false);
}

TEST_CASE("2D primitive intersections preserve mixed precision archive orders",
          "[cpp][spatial][intersects][2d][precision][archive-link]") {
  auto point32 = origin_primitive_2d<float>(tf::cpp::primitive_kind::point);
  auto box64 = origin_primitive_2d<double>(tf::cpp::primitive_kind::aabb);
  CHECK(tf::cpp::intersects(point32, box64).scalar());
  CHECK(tf::cpp::intersects(box64, point32).scalar());

  STATIC_REQUIRE_FALSE(has_intersects<tf::cpp::primitive<float, 2>,
                                      tf::cpp::primitive<float, 3>>::value);
  STATIC_REQUIRE_FALSE(has_intersects<tf::cpp::primitive<double, 3>,
                                      tf::cpp::primitive<double, 2>>::value);
}

TEST_CASE("vectors construct but spatial intersection rejects them",
          "[cpp][spatial][intersects]") {
  auto vector = origin_primitive<float>(tf::cpp::primitive_kind::vector);
  auto point = origin_primitive<float>(tf::cpp::primitive_kind::point);
  const auto owned = triangle_mesh<float>();
  const auto mesh = owned.mesh();
  CHECK_THROWS_AS(tf::cpp::intersects(vector, point), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::intersects(point, vector), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::intersects(mesh, vector), std::invalid_argument);
}

TEST_CASE("primitive intersections preserve scalar batch and broadcast modes",
          "[cpp][spatial][intersects][batch]") {
  auto scalar = origin_primitive<float>(tf::cpp::primitive_kind::point);
  auto points = tf::cpp::primitive<float>(
      tf::cpp::primitive_kind::point,
      make_array<float>({0, 0, 0, 5, 5, 5, 0, 0, 0}, {3, 3}));
  auto matching = tf::cpp::primitive<float>(
      tf::cpp::primitive_kind::aabb,
      make_array<float>(
          {-1, -1, -1, 1, 1, 1, 4, 4, 4, 6, 6, 6, -1, -1, -1, 1, 1, 1},
          {3, 2, 3}));

  CHECK(tf::cpp::intersects(scalar, scalar).is_scalar());
  const auto batch_single = tf::cpp::intersects(points, scalar);
  const auto single_batch = tf::cpp::intersects(scalar, points);
  const auto pairwise = tf::cpp::intersects(points, matching);
  REQUIRE(batch_single.is_batch());
  REQUIRE(single_batch.is_batch());
  REQUIRE(pairwise.is_batch());
  CHECK(batch_single.batch().shape_at(0) == 3);
  CHECK(single_batch.batch().shape_at(0) == 3);
  CHECK(pairwise.batch()[0] == 1);
  CHECK(pairwise.batch()[1] == 1);
  CHECK(pairwise.batch()[2] == 1);

  auto unequal =
      tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                make_array<float>({0, 0, 0, 0, 0, 0}, {2, 3}));
  CHECK_THROWS_AS(tf::cpp::intersects(points, unequal), std::invalid_argument);

  auto empty = tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                         make_empty<float>({0, 3}));
  const auto empty_result = tf::cpp::intersects(empty, scalar);
  REQUIRE(empty_result.is_batch());
  CHECK(empty_result.batch().shape_at(0) == 0);
  CHECK(empty_result.batch().empty());
  CHECK(tf::cpp::intersects(empty, empty).batch().empty());

  auto one = tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                       make_array<float>({0, 0, 0}, {1, 3}));
  const auto one_result = tf::cpp::intersects(one, scalar);
  REQUIRE(one_result.is_batch());
  CHECK(one_result.batch().shape_at(0) == 1);
}

TEST_CASE("primitive mixed precision promotes and results remain unambiguous",
          "[cpp][spatial][intersects][precision]") {
  auto point32 = origin_primitive<float>(tf::cpp::primitive_kind::point);
  auto point64 = origin_primitive<double>(tf::cpp::primitive_kind::point);
  const auto result = tf::cpp::intersects(point32, point64);
  CHECK(result.is_scalar());
  CHECK_FALSE(result.is_batch());
  CHECK(result.scalar());
  CHECK_THROWS_AS(result.batch(), std::logic_error);

  auto batch64 = tf::cpp::primitive<double>(
      tf::cpp::primitive_kind::point,
      make_array<double>({0, 0, 0, 3, 3, 3}, {2, 3}));
  const auto batch_result = tf::cpp::intersects(point32, batch64);
  CHECK(batch_result.is_batch());
  CHECK_THROWS_AS(batch_result.scalar(), std::logic_error);
}

TEST_CASE("mesh and point cloud intersect every runtime spatial kind",
          "[cpp][spatial][intersects][form]") {
  const auto owned_mesh = triangle_mesh<float>();
  const auto owned_cloud = origin_cloud<float>();
  const auto mesh = owned_mesh.mesh();
  const auto cloud = owned_cloud.point_cloud();
  for (const auto kind : spatial_kinds) {
    INFO("form primitive kind " << static_cast<int>(kind));
    auto query = origin_primitive<double>(kind);
    const auto mesh_result = tf::cpp::intersects(mesh, query);
    const auto cloud_result = tf::cpp::intersects(cloud, query);
    CHECK(mesh_result.is_scalar());
    CHECK(cloud_result.is_scalar());
    CHECK(mesh_result.scalar());
    CHECK(cloud_result.scalar());
  }

  auto batch = tf::cpp::primitive<double>(
      tf::cpp::primitive_kind::point,
      make_array<double>({0, 0, 0, 10, 10, 10}, {2, 3}));
  const auto result = tf::cpp::intersects(mesh, batch);
  REQUIRE(result.is_batch());
  CHECK(result.batch()[0] == 1);
  CHECK(result.batch()[1] == 0);
}

TEST_CASE("all form combinations are scalar and reject mixed precision",
          "[cpp][spatial][intersects][form]") {
  const auto owned_a = triangle_mesh<float>();
  const auto owned_b = triangle_mesh<float>();
  const auto owned_cloud_a = origin_cloud<float>();
  const auto owned_cloud_b = origin_cloud<float>();
  const auto mesh_a = owned_a.mesh();
  const auto mesh_b = owned_b.mesh();
  const auto cloud_a = owned_cloud_a.point_cloud();
  const auto cloud_b = owned_cloud_b.point_cloud();

  CHECK(tf::cpp::intersects(mesh_a, mesh_b).scalar());
  CHECK(tf::cpp::intersects(mesh_a, cloud_a).scalar());
  CHECK(tf::cpp::intersects(cloud_a, mesh_a).scalar());
  CHECK(tf::cpp::intersects(cloud_a, cloud_b).scalar());

  const auto owned64 = triangle_mesh<double>();
  const auto owned_cloud64 = origin_cloud<double>();
  const auto mesh64 = owned64.mesh();
  const auto cloud64 = owned_cloud64.point_cloud();
  CHECK_THROWS_AS(tf::cpp::intersects(mesh_a, mesh64), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::intersects(mesh_a, cloud64), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::intersects(cloud_a, mesh64), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::intersects(cloud_a, cloud64), std::invalid_argument);
}

TEST_CASE("intersections reuse fresh trees and rebuild stale trees",
          "[cpp][spatial][intersects][tree]") {
  auto owned = triangle_mesh<float>();
  auto point = origin_primitive<float>(tf::cpp::primitive_kind::point);
  const auto mesh = owned.mesh();
  CHECK_FALSE(owned.cache.is_tree_built());
  CHECK(tf::cpp::intersects(mesh, point).scalar());
  CHECK(owned.cache.is_tree_fresh(mesh.geometry()));
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(tf::cpp::intersects(mesh, point).scalar());
  CHECK(owned.cache.tree_build_count() == 1);

  owned.polygons.points_buffer().data_buffer()[2] = 5;
  owned.cache.points_changed();
  const auto moved = owned.mesh();
  CHECK_FALSE(owned.cache.is_tree_fresh(moved.geometry()));
  static_cast<void>(tf::cpp::intersects(moved, point));
  CHECK(owned.cache.is_tree_fresh(moved.geometry()));
  CHECK(owned.cache.tree_build_count() == 2);

  auto owned_cloud = origin_cloud<float>();
  const auto cloud = owned_cloud.point_cloud();
  CHECK_FALSE(owned_cloud.cache.is_tree_built());
  CHECK(tf::cpp::intersects(cloud, point).scalar());
  CHECK(owned_cloud.cache.tree_build_count() == 1);
  CHECK(tf::cpp::intersects(cloud, point).scalar());
  CHECK(owned_cloud.cache.tree_build_count() == 1);
}

TEST_CASE("async intersects preserve PP scalar batch and mixed precision",
          "[cpp][spatial][intersects][async][primitive]") {
  auto point32 = origin_primitive<float>(tf::cpp::primitive_kind::point);
  auto box64 = origin_primitive<double>(tf::cpp::primitive_kind::aabb);
  auto points64 = tf::cpp::primitive<double>(
      tf::cpp::primitive_kind::point,
      make_array<double>({0, 0, 0, 5, 5, 5}, {2, 3}));

  auto scalar = tf::cpp::async::intersects(point32, box64);
  auto batch = tf::cpp::async::intersects(point32, points64);
  STATIC_REQUIRE(std::is_same_v<decltype(scalar),
                                std::future<tf::cpp::intersection_result>>);
  STATIC_REQUIRE(std::is_same_v<decltype(batch),
                                std::future<tf::cpp::intersection_result>>);
  CHECK(scalar.get().scalar() == tf::cpp::intersects(point32, box64).scalar());
  const auto actual_batch = batch.get().batch();
  const auto expected_batch = tf::cpp::intersects(point32, points64).batch();
  REQUIRE(actual_batch.length() == expected_batch.length());
  CHECK(actual_batch[0] == expected_batch[0]);
  CHECK(actual_batch[1] == expected_batch[1]);

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::intersects(counting_resolver{submissions},
                                           point32, box64);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(custom.get().scalar());

  auto owned_a = point32;
  auto owned_b = box64;
  auto owned = tf::cpp::async::intersects(owned_a, owned_b);
  owned_a = tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                      make_array<float>({5, 5, 5}, {3}));
  owned_b = tf::cpp::primitive<double>(tf::cpp::primitive_kind::point,
                                       make_array<double>({10, 10, 10}, {3}));
  CHECK(owned.get().scalar());
}

TEST_CASE("async intersects preserve FP FF cache empty and error behavior",
          "[cpp][spatial][intersects][async][form]") {
  auto owned = triangle_mesh<float>();
  auto owned_cloud = origin_cloud<float>();
  const auto mesh = owned.mesh();
  const auto cloud = owned_cloud.point_cloud();
  auto point64 = origin_primitive<double>(tf::cpp::primitive_kind::point);
  auto points64 = tf::cpp::primitive<double>(
      tf::cpp::primitive_kind::point,
      make_array<double>({0, 0, 0, 5, 5, 5}, {2, 3}));

  // a cache shared by concurrent jobs is filled before it is shared
  tf::cpp::build_tree(mesh);
  tf::cpp::build_tree(cloud);
  const auto mesh_builds = owned.cache.tree_build_count();
  const auto cloud_builds = owned_cloud.cache.tree_build_count();

  auto fp = tf::cpp::async::intersects(mesh, point64);
  auto fp_batch = tf::cpp::async::intersects(cloud, points64);
  auto ff = tf::cpp::async::intersects(mesh, cloud);
  STATIC_REQUIRE(
      std::is_same_v<decltype(fp), std::future<tf::cpp::intersection_result>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(ff), std::future<tf::cpp::intersection_result>>);
  CHECK(fp.get().scalar());
  const auto values = fp_batch.get().batch();
  CHECK(values[0] == 1);
  CHECK(values[1] == 0);
  CHECK(ff.get().scalar());
  CHECK(tf::cpp::async::intersects(mesh, cloud).get().scalar());
  CHECK(owned.cache.tree_build_count() == mesh_builds);
  CHECK(owned_cloud.cache.tree_build_count() == cloud_builds);

  auto held_query = point64;
  auto held = tf::cpp::async::intersects(
      held_carrier_of(triangle_mesh<float>()), held_query);
  held_query = tf::cpp::primitive<double>(tf::cpp::primitive_kind::point,
                                          make_array<double>({5, 5, 5}, {3}));
  CHECK(held.get().scalar());

  auto empty = tf::cpp::primitive<double>(tf::cpp::primitive_kind::point,
                                          make_empty<double>({0, 3}));
  const auto empty_result = tf::cpp::async::intersects(mesh, empty).get();
  REQUIRE(empty_result.is_batch());
  CHECK(empty_result.batch().empty());

  auto vector = origin_primitive<float>(tf::cpp::primitive_kind::vector);
  auto failure = tf::cpp::async::intersects(mesh, vector);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);

  const auto owned64 = triangle_mesh<double>();
  auto mixed_failure = tf::cpp::async::intersects(mesh, owned64.mesh());
  CHECK_THROWS_AS(mixed_failure.get(), std::invalid_argument);
}

#include "./test_intersects_matrix.hpp"
