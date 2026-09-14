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
#include "deterministic_points.hpp"

#include "trueform/cpp/spatial/closest_metric_point_pair.hpp"
#include "trueform/cpp/spatial/distance.hpp"
#include "trueform/cpp/spatial/intersects.hpp"
#include "trueform/cpp/spatial/ray_cast.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <initializer_list>
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

template <typename T, std::size_t Dims = 3>
auto primitive(tf::cpp::primitive_kind kind, std::initializer_list<T> values,
               tf::small_vector<int, 3> shape) -> tf::cpp::primitive<T, Dims> {
  return tf::cpp::primitive<T, Dims>(kind,
                                     make_array<T>(values, std::move(shape)));
}

template <typename T> auto point(T x, T y, T z) -> tf::cpp::primitive<T> {
  return primitive<T>(tf::cpp::primitive_kind::point, {x, y, z}, {3});
}

template <typename T> auto unit_box() -> tf::cpp::primitive<T> {
  return primitive<T>(tf::cpp::primitive_kind::aabb,
                      {T{0}, T{0}, T{0}, T{1}, T{1}, T{1}}, {2, 3});
}

template <typename T> auto square() -> tf::cpp::primitive<T> {
  return primitive<T>(
      tf::cpp::primitive_kind::polygon,
      {T{0}, T{0}, T{0}, T{1}, T{0}, T{0}, T{1}, T{1}, T{0}, T{0}, T{1}, T{0}},
      {4, 3});
}

template <typename T>
auto require_shape(const tf::cpp::nd_array<T> &array,
                   std::initializer_list<int> shape) -> void {
  REQUIRE(array.ndim() == static_cast<int>(shape.size()));
  int dimension = 0;
  for (const auto extent : shape)
    CHECK(array.shape_at(dimension++) == extent);
}

template <typename T> constexpr auto margin() -> double {
  return std::is_same<T, float>::value ? 1e-5 : 1e-11;
}

template <typename Real, std::size_t Dims>
auto require_ray_hit_t(const tf::cpp::primitive<Real, Dims> &ray,
                       const tf::cpp::primitive<Real, Dims> &target,
                       Real expected_t)
    -> tf::cpp::ray_cast_result<tf::cpp::default_index_t, Real> {
  const auto result = tf::cpp::ray_cast(ray, target);
  REQUIRE(result.is_scalar());
  REQUIRE(result.scalar().hit);
  CHECK(result.scalar().t == expected_t);
  return result.scalar();
}

template <typename Real, std::size_t Dims>
auto require_ray_miss(const tf::cpp::primitive<Real, Dims> &ray,
                      const tf::cpp::primitive<Real, Dims> &target) -> void {
  const auto result = tf::cpp::ray_cast(ray, target);
  REQUIRE(result.is_scalar());
  CHECK_FALSE(result.scalar().hit);
}

} // namespace

TEMPLATE_TEST_CASE(
    "Python distance fixtures preserve signed plane and clamped segment values",
    "[cpp][spatial][python-parity][distance]", float, double) {
  const auto plane =
      primitive<TestType>(tf::cpp::primitive_kind::plane, {0, 0, 1, 0}, {4});
  const auto plane_queries = primitive<TestType>(
      tf::cpp::primitive_kind::point,
      {0, 0, 2, 1, 1, 0, TestType{0.5}, TestType{0.5}, -1, 2, 2, 3}, {4, 3});
  const auto signed_distances = tf::cpp::distance(plane_queries, plane);
  REQUIRE(signed_distances.is_batch());
  const auto signed_values = signed_distances.batch();
  require_shape(signed_values, {4});
  CHECK(signed_values[0] == Catch::Approx(2).margin(margin<TestType>()));
  CHECK(signed_values[1] == Catch::Approx(0).margin(margin<TestType>()));
  CHECK(signed_values[2] == Catch::Approx(-1).margin(margin<TestType>()));
  CHECK(signed_values[3] == Catch::Approx(3).margin(margin<TestType>()));

  const auto segment = primitive<TestType>(tf::cpp::primitive_kind::segment,
                                           {0, 0, 0, 2, 0, 0}, {2, 3});
  const auto segment_queries = primitive<TestType>(
      tf::cpp::primitive_kind::point,
      {1, 0, 0, 1, 1, 0, 3, 0, 0, -1, 0, 0, 1, 2, 0}, {5, 3});
  const auto distances = tf::cpp::distance(segment_queries, segment).batch();
  const auto distances2 = tf::cpp::distance2(segment_queries, segment).batch();
  require_shape(distances, {5});
  require_shape(distances2, {5});
  const TestType expected[]{0, 1, 1, 1, 2};
  for (std::size_t index = 0; index < 5; ++index) {
    CHECK(distances[index] ==
          Catch::Approx(expected[index]).margin(margin<TestType>()));
    CHECK(distances2[index] == Catch::Approx(expected[index] * expected[index])
                                   .margin(margin<TestType>()));
  }

  const auto one_query =
      primitive<TestType>(tf::cpp::primitive_kind::point, {1, 1, 0}, {1, 3});
  const auto one_result = tf::cpp::distance(one_query, segment);
  REQUIRE(one_result.is_batch());
  require_shape(one_result.batch(), {1});
  CHECK(one_result.batch()[0] == Catch::Approx(1).margin(margin<TestType>()));

  constexpr int large_batch_size = 1000;
  const auto large_points = tf::cpp::test::spiral_points<TestType>(
      large_batch_size, TestType{2}, TestType{3}, TestType{4});
  const auto large_queries = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::point, large_points);
  const auto large_distances = tf::cpp::distance(large_queries, plane).batch();
  require_shape(large_distances, {large_batch_size});
  for (int index = 0; index < large_batch_size; ++index)
    CHECK(large_distances[static_cast<std::size_t>(index)] ==
          Catch::Approx(large_points[static_cast<std::size_t>(index) * 3 + 2])
              .margin(margin<TestType>()));
}

TEMPLATE_TEST_CASE(
    "Python scalar distance fixtures preserve containment and operand symmetry",
    "[cpp][spatial][python-parity][distance]", float, double) {
  const auto box = unit_box<TestType>();
  const auto inside =
      point<TestType>(TestType{0.5}, TestType{0.5}, TestType{0.5});
  const auto outside = point<TestType>(2, 2, 1);

  CHECK(tf::cpp::distance(inside, box).scalar() ==
        Catch::Approx(0).margin(margin<TestType>()));
  CHECK(tf::cpp::distance2(outside, box).scalar() ==
        Catch::Approx(2).margin(margin<TestType>()));
  CHECK(tf::cpp::distance(box, outside).scalar() ==
        Catch::Approx(1.4142135623730951).margin(margin<TestType>()));

  const auto polygon = square<TestType>();
  CHECK(tf::cpp::distance(point<TestType>(TestType{0.5}, TestType{0.5}, 0),
                          polygon)
            .scalar() == Catch::Approx(0).margin(margin<TestType>()));
  CHECK(tf::cpp::distance2(point<TestType>(2, 2, 0), polygon).scalar() ==
        Catch::Approx(2).margin(margin<TestType>()));
}

TEMPLATE_TEST_CASE("Python intersection fixtures distinguish inside miss "
                   "boundary and crossing",
                   "[cpp][spatial][python-parity][intersects]", float, double) {
  const auto box = unit_box<TestType>();
  const auto points = primitive<TestType>(
      tf::cpp::primitive_kind::point,
      {TestType{0.5}, TestType{0.5}, TestType{0.5}, 2, 2, 2, 1, 1, 1}, {3, 3});
  const auto point_results = tf::cpp::intersects(points, box);
  REQUIRE(point_results.is_batch());
  const auto point_values = point_results.batch();
  require_shape(point_values, {3});
  CHECK(point_values[0] == 1);
  CHECK(point_values[1] == 0);
  CHECK(point_values[2] == 1);

  const auto segments = primitive<TestType>(
      tf::cpp::primitive_kind::segment,
      {TestType{0.5}, TestType{0.5}, -1, TestType{0.5}, TestType{0.5}, 2, 2,
       TestType{0.5}, TestType{0.5}, 3, TestType{0.5}, TestType{0.5}},
      {2, 2, 3});
  const auto segment_results = tf::cpp::intersects(segments, box);
  REQUIRE(segment_results.is_batch());
  const auto segment_values = segment_results.batch();
  require_shape(segment_values, {2});
  CHECK(segment_values[0] == 1);
  CHECK(segment_values[1] == 0);
  CHECK(tf::cpp::intersects(box, segments).batch()[0] == 1);
}

TEMPLATE_TEST_CASE(
    "Python ray cast fixtures expose synchronized hit miss and touch results",
    "[cpp][spatial][python-parity][ray-cast]", float, double) {
  const auto rays =
      primitive<TestType>(tf::cpp::primitive_kind::ray,
                          {0, TestType{0.5}, 0, 1, 0, 0, 0, TestType{0.5}, 0,
                           -1, 0, 0, 0, 1, 0, 1, 0, 0},
                          {3, 2, 3});
  const auto segment = primitive<TestType>(tf::cpp::primitive_kind::segment,
                                           {1, 0, 0, 1, 1, 0}, {2, 3});
  const auto result = tf::cpp::ray_cast(rays, segment);
  REQUIRE(result.is_batch());
  const auto &values = result.batch();
  require_shape(values.hits, {3});
  require_shape(values.ts, {3});
  CHECK(values.hits[0] == 1);
  CHECK(values.ts[0] == Catch::Approx(1).margin(margin<TestType>()));
  CHECK(values.hits[1] == 0);
  CHECK(values.hits[2] == 1);
  CHECK(values.ts[2] == Catch::Approx(1).margin(margin<TestType>()));

  tf::cpp::ray_cast_options<TestType> exact_touch;
  exact_touch.min_t = 1;
  exact_touch.max_t = 1;
  const auto touch = tf::cpp::ray_cast(rays.at(0), segment, exact_touch);
  REQUIRE(touch.is_scalar());
  CHECK(touch.scalar().hit);
  CHECK(touch.scalar().t == Catch::Approx(1).margin(margin<TestType>()));

  tf::cpp::ray_cast_options<TestType> too_short;
  too_short.max_t = TestType{0.5};
  CHECK_FALSE(tf::cpp::ray_cast(rays.at(0), segment, too_short).scalar().hit);
}

TEMPLATE_TEST_CASE(
    "Python 3D ray-plane fixtures preserve hits misses and hit points",
    "[cpp][spatial][python-parity][ray-cast][plane]", float, double) {
  const auto plane =
      primitive<TestType>(tf::cpp::primitive_kind::plane, {0, 0, 1, 0}, {4});
  const auto hit =
      primitive<TestType>(tf::cpp::primitive_kind::ray,
                          {TestType{0.5}, TestType{0.5}, 2, 0, 0, -1}, {2, 3});
  const auto hit_result = require_ray_hit_t(hit, plane, TestType{2});
  const TestType origin[]{TestType{0.5}, TestType{0.5}, TestType{2}};
  const TestType direction[]{TestType{0}, TestType{0}, TestType{-1}};
  const TestType expected_point[]{TestType{0.5}, TestType{0.5}, TestType{0}};
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    CHECK(origin[dimension] + hit_result.t * direction[dimension] ==
          expected_point[dimension]);

  const auto away =
      primitive<TestType>(tf::cpp::primitive_kind::ray,
                          {TestType{0.5}, TestType{0.5}, 2, 0, 0, 1}, {2, 3});
  const auto parallel =
      primitive<TestType>(tf::cpp::primitive_kind::ray,
                          {TestType{0.5}, TestType{0.5}, 2, 1, 0, 0}, {2, 3});
  require_ray_miss(away, plane);
  require_ray_miss(parallel, plane);
}

TEMPLATE_TEST_CASE(
    "Python 2D and 3D ray-polygon fixtures preserve boundary parameters",
    "[cpp][spatial][python-parity][ray-cast][polygon]", float, double) {
  const auto square = primitive<TestType, 2>(tf::cpp::primitive_kind::polygon,
                                             {0, 0, 1, 0, 1, 1, 0, 1}, {4, 2});
  const auto hit_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                             {-1, TestType{0.5}, 1, 0}, {2, 2});
  const auto miss_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                              {2, TestType{0.5}, 1, 0}, {2, 2});
  const auto inside_2d =
      primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                             {TestType{0.5}, TestType{0.5}, 1, 0}, {2, 2});
  require_ray_hit_t(hit_2d, square, TestType{1});
  require_ray_miss(miss_2d, square);
  require_ray_hit_t(inside_2d, square, TestType{0.5});

  const auto triangle =
      primitive<TestType>(tf::cpp::primitive_kind::polygon,
                          {0, 0, 0, 1, 0, 0, TestType{0.5}, 1, 0}, {3, 3});
  const auto hit_3d =
      primitive<TestType>(tf::cpp::primitive_kind::ray,
                          {TestType{0.5}, TestType{0.3}, 2, 0, 0, -1}, {2, 3});
  const auto miss_3d = primitive<TestType>(tf::cpp::primitive_kind::ray,
                                           {2, 2, 2, 0, 0, -1}, {2, 3});
  require_ray_hit_t(hit_3d, triangle, TestType{2});
  require_ray_miss(miss_3d, triangle);
}

TEMPLATE_TEST_CASE(
    "Python 2D and 3D ray-segment fixtures preserve hit parameters",
    "[cpp][spatial][python-parity][ray-cast][segment]", float, double) {
  const auto segment_2d = primitive<TestType, 2>(
      tf::cpp::primitive_kind::segment, {1, 0, 1, 2}, {2, 2});
  const auto hit_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                             {0, 1, 1, 0}, {2, 2});
  const auto miss_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                              {0, 3, 1, 0}, {2, 2});
  const auto away_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                              {0, 1, -1, 0}, {2, 2});
  require_ray_hit_t(hit_2d, segment_2d, TestType{1});
  require_ray_miss(miss_2d, segment_2d);
  require_ray_miss(away_2d, segment_2d);

  const auto segment_3d = primitive<TestType>(
      tf::cpp::primitive_kind::segment,
      {0, TestType{0.5}, TestType{0.5}, 2, TestType{0.5}, TestType{0.5}},
      {2, 3});
  const auto hit_3d = primitive<TestType>(
      tf::cpp::primitive_kind::ray, {1, 0, TestType{0.5}, 0, 1, 0}, {2, 3});
  const auto miss_3d =
      primitive<TestType>(tf::cpp::primitive_kind::ray,
                          {0, TestType{1.5}, TestType{0.5}, 1, 0, 0}, {2, 3});
  require_ray_hit_t(hit_3d, segment_3d, TestType{0.5});
  require_ray_miss(miss_3d, segment_3d);
}

TEMPLATE_TEST_CASE(
    "Python 2D and 3D ray-line fixtures preserve parallel and skew misses",
    "[cpp][spatial][python-parity][ray-cast][line]", float, double) {
  const auto line_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::line,
                                              {1, 0, 0, 1}, {2, 2});
  const auto hit_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                             {0, TestType{0.5}, 1, 0}, {2, 2});
  const auto parallel_2d = primitive<TestType, 2>(
      tf::cpp::primitive_kind::ray, {0, TestType{0.5}, 0, 1}, {2, 2});
  const auto away_2d = primitive<TestType, 2>(
      tf::cpp::primitive_kind::ray, {0, TestType{0.5}, -1, 0}, {2, 2});
  require_ray_hit_t(hit_2d, line_2d, TestType{1});
  require_ray_miss(parallel_2d, line_2d);
  require_ray_miss(away_2d, line_2d);

  const auto line_3d = primitive<TestType>(tf::cpp::primitive_kind::line,
                                           {0, 0, 0, 0, 0, 1}, {2, 3});
  const auto hit_3d = primitive<TestType>(
      tf::cpp::primitive_kind::ray, {1, 0, TestType{0.5}, -1, 0, 0}, {2, 3});
  const auto skew_3d = primitive<TestType>(tf::cpp::primitive_kind::ray,
                                           {1, 1, 0, 1, 0, 0}, {2, 3});
  require_ray_hit_t(hit_3d, line_3d, TestType{1});
  require_ray_miss(skew_3d, line_3d);
}

TEMPLATE_TEST_CASE(
    "Python 2D and 3D ray-AABB fixtures preserve entry and inside parameters",
    "[cpp][spatial][python-parity][ray-cast][aabb]", float, double) {
  const auto box_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::aabb,
                                             {0, 0, 1, 1}, {2, 2});
  const auto hit_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                             {-1, TestType{0.5}, 1, 0}, {2, 2});
  const auto miss_2d = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                              {2, TestType{0.5}, 1, 0}, {2, 2});
  const auto inside_2d =
      primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                             {TestType{0.5}, TestType{0.5}, 1, 0}, {2, 2});
  require_ray_hit_t(hit_2d, box_2d, TestType{1});
  require_ray_miss(miss_2d, box_2d);
  require_ray_hit_t(inside_2d, box_2d, TestType{0});

  const auto box_3d = primitive<TestType>(tf::cpp::primitive_kind::aabb,
                                          {0, 0, 0, 1, 1, 1}, {2, 3});
  const auto hit_3d =
      primitive<TestType>(tf::cpp::primitive_kind::ray,
                          {TestType{0.5}, TestType{0.5}, 2, 0, 0, -1}, {2, 3});
  const auto miss_3d = primitive<TestType>(tf::cpp::primitive_kind::ray,
                                           {2, 2, 2, 0, 0, -1}, {2, 3});
  const auto away_3d =
      primitive<TestType>(tf::cpp::primitive_kind::ray,
                          {TestType{0.5}, TestType{0.5}, 2, 0, 0, 1}, {2, 3});
  require_ray_hit_t(hit_3d, box_3d, TestType{1});
  require_ray_miss(miss_3d, box_3d);
  require_ray_miss(away_3d, box_3d);
}

TEMPLATE_TEST_CASE("Python closest pair fixtures preserve projections operand "
                   "order and shapes",
                   "[cpp][spatial][python-parity][closest-pair]", float,
                   double) {
  const auto plane =
      primitive<TestType>(tf::cpp::primitive_kind::plane, {0, 0, 1, 0}, {4});
  const auto above = point<TestType>(1, 2, 5);
  const auto forward =
      std::get<tf::cpp::closest_metric_point_pair_result<TestType>>(
          tf::cpp::closest_metric_point_pair(above, plane));
  CHECK(forward.distance2 == Catch::Approx(25).margin(margin<TestType>()));
  require_shape(forward.point0, {3});
  require_shape(forward.point1, {3});
  CHECK(forward.point0[0] == Catch::Approx(1));
  CHECK(forward.point0[1] == Catch::Approx(2));
  CHECK(forward.point0[2] == Catch::Approx(5));
  CHECK(forward.point1[0] == Catch::Approx(1));
  CHECK(forward.point1[1] == Catch::Approx(2));
  CHECK(forward.point1[2] == Catch::Approx(0));

  const auto reverse =
      std::get<tf::cpp::closest_metric_point_pair_result<TestType>>(
          tf::cpp::closest_metric_point_pair(plane, above));
  CHECK(reverse.distance2 == Catch::Approx(forward.distance2));
  CHECK(reverse.point0[2] == Catch::Approx(0));
  CHECK(reverse.point1[2] == Catch::Approx(5));

  const auto queries = primitive<TestType>(
      tf::cpp::primitive_kind::point, {0, 1, 0, TestType{0.5}, 2, 0}, {2, 3});
  const auto segment = primitive<TestType>(tf::cpp::primitive_kind::segment,
                                           {0, 0, 0, 1, 0, 0}, {2, 3});
  const auto batch =
      std::get<tf::cpp::closest_metric_point_pair_batch_result<TestType>>(
          tf::cpp::closest_metric_point_pair(queries, segment));
  require_shape(batch.distances, {2});
  require_shape(batch.points0, {2, 3});
  require_shape(batch.points1, {2, 3});
  CHECK(batch.distances[0] == Catch::Approx(1));
  CHECK(batch.distances[1] == Catch::Approx(4));
  CHECK(batch.points0[3] == Catch::Approx(TestType{0.5}));
  CHECK(batch.points0[4] == Catch::Approx(2));
  CHECK(batch.points1[3] == Catch::Approx(TestType{0.5}));
  CHECK(batch.points1[4] == Catch::Approx(0));
}

TEMPLATE_TEST_CASE(
    "Python closest pair polygon and touching box fixtures remain geometric",
    "[cpp][spatial][python-parity][closest-pair]", float, double) {
  const auto polygon =
      primitive<TestType>(tf::cpp::primitive_kind::polygon,
                          {0, 0, 0, 1, 0, 0, TestType{0.5}, 1, 0}, {3, 3});
  const auto above = point<TestType>(TestType{0.5}, TestType{0.3}, 2);
  const auto polygon_pair =
      std::get<tf::cpp::closest_metric_point_pair_result<TestType>>(
          tf::cpp::closest_metric_point_pair(above, polygon));
  CHECK(polygon_pair.distance2 == Catch::Approx(4).margin(margin<TestType>()));
  CHECK(polygon_pair.point0[2] == Catch::Approx(2));
  CHECK(polygon_pair.point1[0] == Catch::Approx(TestType{0.5}));
  CHECK(polygon_pair.point1[1] == Catch::Approx(TestType{0.3}));
  CHECK(polygon_pair.point1[2] == Catch::Approx(0));

  const auto box0 = unit_box<TestType>();
  const auto touching_box = primitive<TestType>(tf::cpp::primitive_kind::aabb,
                                                {1, 0, 0, 2, 1, 1}, {2, 3});
  const auto touching_pair =
      std::get<tf::cpp::closest_metric_point_pair_result<TestType>>(
          tf::cpp::closest_metric_point_pair(box0, touching_box));
  CHECK(touching_pair.distance2 == Catch::Approx(0).margin(margin<TestType>()));
}

namespace {

template <typename Real, std::size_t Dims>
auto geometric_primitive(
    tf::cpp::primitive_kind kind,
    std::initializer_list<std::array<Real, 3>> control_points)
    -> tf::cpp::primitive<Real, Dims> {
  tf::buffer<Real> values;
  values.allocate(control_points.size() * Dims);
  auto output = values.begin();
  for (const auto &control_point : control_points)
    for (std::size_t dimension = 0; dimension < Dims; ++dimension)
      *output++ = control_point[dimension];

  tf::small_vector<int, 3> shape;
  if (kind == tf::cpp::primitive_kind::point)
    shape = {static_cast<int>(Dims)};
  else
    shape = {static_cast<int>(control_points.size()), static_cast<int>(Dims)};
  return tf::cpp::primitive<Real, Dims>(
      kind, tf::cpp::nd_array<Real>::from_buffer(std::move(values),
                                                 std::move(shape)));
}

template <typename Real, std::size_t Dims>
auto point_d(Real x, Real y, Real z = Real{})
    -> tf::cpp::primitive<Real, Dims> {
  return geometric_primitive<Real, Dims>(tf::cpp::primitive_kind::point,
                                         {{x, y, z}});
}

template <typename Real>
auto plane_d(Real nx, Real ny, Real nz, Real d) -> tf::cpp::primitive<Real, 3> {
  return primitive<Real>(tf::cpp::primitive_kind::plane, {nx, ny, nz, d}, {4});
}

template <typename Real, std::size_t Dims>
auto closest_pair(const tf::cpp::primitive<Real, Dims> &a,
                  const tf::cpp::primitive<Real, Dims> &b)
    -> tf::cpp::closest_metric_point_pair_result<Real> {
  return std::get<tf::cpp::closest_metric_point_pair_result<Real>>(
      tf::cpp::closest_metric_point_pair(a, b));
}

template <typename Real> auto check_scalar(Real actual, Real expected) -> void {
  CHECK(actual == Catch::Approx(expected).margin(margin<Real>()));
}

template <typename Real, std::size_t Dims>
auto check_point(const tf::cpp::nd_array<Real> &actual,
                 const std::array<Real, 3> &expected) -> void {
  require_shape(actual, {static_cast<int>(Dims)});
  for (std::size_t dimension = 0; dimension < Dims; ++dimension)
    CHECK(actual[dimension] ==
          Catch::Approx(expected[dimension]).margin(margin<Real>()));
}

template <typename Real, std::size_t Dims>
auto check_points_equal(const tf::cpp::nd_array<Real> &a,
                        const tf::cpp::nd_array<Real> &b) -> void {
  require_shape(a, {static_cast<int>(Dims)});
  require_shape(b, {static_cast<int>(Dims)});
  for (std::size_t dimension = 0; dimension < Dims; ++dimension)
    CHECK(a[dimension] == Catch::Approx(b[dimension]).margin(margin<Real>()));
}

template <typename Real, std::size_t Dims>
auto check_distance2_matches(const tf::cpp::primitive<Real, Dims> &a,
                             const tf::cpp::primitive<Real, Dims> &b,
                             Real expected) -> void {
  const auto distance = tf::cpp::distance2(a, b);
  REQUIRE(distance.is_scalar());
  check_scalar(distance.scalar(), expected);
}

template <typename Left, typename Right, typename = void>
struct is_closest_pair_invocable : std::false_type {};

template <typename Left, typename Right>
struct is_closest_pair_invocable<
    Left, Right,
    std::void_t<decltype(tf::cpp::closest_metric_point_pair(
        std::declval<const Left &>(), std::declval<const Right &>()))>>
    : std::true_type {};

auto python_polygon_basic_parity() -> void {
  {
    INFO("test_point_polygon_2d_inside[float32]");
    const auto polygon = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    const auto query = point_d<float, 2>(0.5F, 0.5F);
    check_scalar(closest_pair(query, polygon).distance2, 0.0F);
  }
  {
    INFO("test_point_polygon_2d_outside[float32]");
    const auto polygon = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    const auto query = point_d<float, 2>(2.0F, 0.5F);
    const auto result = closest_pair(query, polygon);
    check_scalar(result.distance2, 1.0F);
    check_point<float, 2>(result.point1, {1.0F, 0.5F, 0.0F});
  }
  {
    INFO("test_point_polygon_3d_inside[float64]");
    const auto polygon = geometric_primitive<double, 3>(
        tf::cpp::primitive_kind::polygon, {{0, 0, 0}, {1, 0, 0}, {0.5, 1, 0}});
    check_scalar(
        closest_pair(point_d<double, 3>(0.5, 0.3, 0.0), polygon).distance2,
        0.0);
  }
  {
    INFO("test_point_polygon_3d_above[float64]");
    const auto polygon = geometric_primitive<double, 3>(
        tf::cpp::primitive_kind::polygon, {{0, 0, 0}, {1, 0, 0}, {0.5, 1, 0}});
    const auto result =
        closest_pair(point_d<double, 3>(0.5, 0.3, 2.0), polygon);
    check_scalar(result.distance2, 4.0);
    check_point<double, 3>(result.point1, {0.5, 0.3, 0.0});
  }
  {
    INFO("test_polygon_polygon_2d_separate[float32]");
    const auto a = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    const auto b = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{2, 0, 0}, {3, 0, 0}, {3, 1, 0}, {2, 1, 0}});
    check_scalar(closest_pair(a, b).distance2, 1.0F);
  }
  {
    INFO("test_polygon_polygon_2d_overlapping[float32]");
    const auto a = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    const auto b = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{0.5F, 0.5F, 0}, {1.5F, 0.5F, 0}, {1.5F, 1.5F, 0}, {0.5F, 1.5F, 0}});
    check_scalar(closest_pair(a, b).distance2, 0.0F);
  }
  {
    INFO("test_polygon_polygon_3d[float64]");
    const auto a = geometric_primitive<double, 3>(
        tf::cpp::primitive_kind::polygon, {{0, 0, 0}, {1, 0, 0}, {0.5, 1, 0}});
    const auto b = geometric_primitive<double, 3>(
        tf::cpp::primitive_kind::polygon, {{0, 0, 2}, {1, 0, 2}, {0.5, 1, 2}});
    check_scalar(closest_pair(a, b).distance2, 4.0);
  }
  {
    INFO("test_segment_polygon_2d_intersecting[float32]");
    const auto segment = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::segment, {{0.5F, -0.5F, 0}, {0.5F, 1.5F, 0}});
    const auto polygon = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    check_scalar(closest_pair(segment, polygon).distance2, 0.0F);
  }
  {
    INFO("test_segment_polygon_2d_outside[float32]");
    const auto segment = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::segment, {{2, 0, 0}, {3, 0, 0}});
    const auto polygon = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    check_scalar(closest_pair(segment, polygon).distance2, 1.0F);
  }
  {
    INFO("test_ray_polygon_3d_hitting[float32]");
    const auto ray = geometric_primitive<float, 3>(
        tf::cpp::primitive_kind::ray,
        {{0.5F, 0.3F, 2.0F}, {0.0F, 0.0F, -1.0F}});
    const auto polygon = geometric_primitive<float, 3>(
        tf::cpp::primitive_kind::polygon, {{0, 0, 0}, {1, 0, 0}, {0.5F, 1, 0}});
    check_scalar(closest_pair(ray, polygon).distance2, 0.0F);
  }
  {
    INFO("test_ray_polygon_3d_missing[float32]");
    const auto ray = geometric_primitive<float, 3>(
        tf::cpp::primitive_kind::ray, {{0.5F, 0.3F, 2.0F}, {0.0F, 0.0F, 1.0F}});
    const auto polygon = geometric_primitive<float, 3>(
        tf::cpp::primitive_kind::polygon, {{0, 0, 0}, {1, 0, 0}, {0.5F, 1, 0}});
    check_scalar(closest_pair(ray, polygon).distance2, 4.0F);
  }
  {
    INFO("test_line_polygon_2d_intersecting[float32]");
    const auto line = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::line, {{0.5F, -1.0F, 0}, {0.0F, 1.0F, 0}});
    const auto polygon = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    check_scalar(closest_pair(line, polygon).distance2, 0.0F);
  }
  {
    INFO("test_line_polygon_2d_parallel[float32]");
    const auto line = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::line, {{2, 0, 0}, {0, 1, 0}});
    const auto polygon = geometric_primitive<float, 2>(
        tf::cpp::primitive_kind::polygon,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    check_scalar(closest_pair(line, polygon).distance2, 1.0F);
  }

  using primitive2 = tf::cpp::primitive<float, 2>;
  using primitive3 = tf::cpp::primitive<float, 3>;
  static_assert(
      !is_closest_pair_invocable<primitive2, primitive3>::value,
      "test_dimension_mismatch_point must be rejected at compile time");
  static_assert(
      !is_closest_pair_invocable<primitive3, primitive2>::value,
      "test_dimension_mismatch_polygon must be rejected at compile time");
}

template <typename Real, std::size_t Dims>
auto python_metric_primitive_parity() -> void {
  INFO("dtype=" << (std::is_same<Real, float>::value ? "float32" : "float64")
                << ", dims=" << Dims);
  {
    INFO("test_point_point_separated");
    const auto a = point_d<Real, Dims>(0, 0, 0);
    const auto b = point_d<Real, Dims>(3, 0, 0);
    const auto result = closest_pair(a, b);
    check_scalar(result.distance2, Real{9});
    check_point<Real, Dims>(result.point0, {Real{0}, Real{0}, Real{0}});
    check_point<Real, Dims>(result.point1, {Real{3}, Real{0}, Real{0}});
    check_distance2_matches(a, b, result.distance2);
  }
  {
    INFO("test_point_segment_perpendicular");
    const auto segment = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment, {{0, 0, 0}, {4, 0, 0}});
    const auto query = point_d<Real, Dims>(2, 3, 0);
    const auto result = closest_pair(query, segment);
    check_scalar(result.distance2, Real{9});
    check_point<Real, Dims>(result.point0, {Real{2}, Real{3}, Real{0}});
    check_point<Real, Dims>(result.point1, {Real{2}, Real{0}, Real{0}});
    check_distance2_matches(query, segment, result.distance2);
  }
  {
    INFO("test_point_segment_endpoint");
    const auto segment = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment, {{0, 0, 0}, {4, 0, 0}});
    const auto query = point_d<Real, Dims>(-2, 0, 0);
    const auto result = closest_pair(query, segment);
    check_scalar(result.distance2, Real{4});
    check_point<Real, Dims>(result.point1, {Real{0}, Real{0}, Real{0}});
    check_distance2_matches(query, segment, result.distance2);
  }
  {
    INFO("test_point_ray_perpendicular");
    const auto ray = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::ray, {{0, 0, 0}, {1, 0, 0}});
    const auto query = point_d<Real, Dims>(3, 4, 0);
    const auto result = closest_pair(query, ray);
    check_scalar(result.distance2, Real{16});
    check_point<Real, Dims>(result.point1, {Real{3}, Real{0}, Real{0}});
    check_distance2_matches(query, ray, result.distance2);
  }
  {
    INFO("test_point_ray_behind_origin");
    const auto ray = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::ray, {{0, 0, 0}, {1, 0, 0}});
    const auto query = point_d<Real, Dims>(-3, 4, 0);
    const auto result = closest_pair(query, ray);
    check_scalar(result.distance2, Real{25});
    check_point<Real, Dims>(result.point1, {Real{0}, Real{0}, Real{0}});
    check_distance2_matches(query, ray, result.distance2);
  }
  {
    INFO("test_point_line_perpendicular");
    const auto line = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::line, {{0, 0, 0}, {1, 0, 0}});
    const auto query = point_d<Real, Dims>(5, 12, 0);
    const auto result = closest_pair(query, line);
    check_scalar(result.distance2, Real{144});
    check_point<Real, Dims>(result.point1, {Real{5}, Real{0}, Real{0}});
    check_distance2_matches(query, line, result.distance2);
  }
  {
    INFO("test_segment_segment_parallel");
    const auto a = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment, {{0, 0, 0}, {4, 0, 0}});
    const auto b = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment, {{0, 3, 0}, {4, 3, 0}});
    const auto result = closest_pair(a, b);
    check_scalar(result.distance2, Real{9});
    check_scalar(result.point0[0], result.point1[0]);
    check_distance2_matches(a, b, result.distance2);
  }
  {
    INFO("test_segment_segment_endpoint_to_midpoint");
    const auto a = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment, {{0, 0, 0}, {4, 0, 0}});
    const auto b = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment, {{2, 3, 0}, {2, 6, 0}});
    const auto result = closest_pair(a, b);
    check_scalar(result.distance2, Real{9});
    check_point<Real, Dims>(result.point0, {Real{2}, Real{0}, Real{0}});
    check_point<Real, Dims>(result.point1, {Real{2}, Real{3}, Real{0}});
    check_distance2_matches(a, b, result.distance2);
  }
  {
    INFO("test_segment_ray_separated");
    const auto segment = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment, {{0, 0, 0}, {2, 0, 0}});
    const auto ray = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::ray, {{4, 3, 0}, {0, 1, 0}});
    const auto result = closest_pair(segment, ray);
    check_scalar(result.distance2, Real{13});
    check_point<Real, Dims>(result.point0, {Real{2}, Real{0}, Real{0}});
    check_point<Real, Dims>(result.point1, {Real{4}, Real{3}, Real{0}});
    check_distance2_matches(segment, ray, result.distance2);
  }
  {
    INFO("test_segment_line_perpendicular");
    const auto segment = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment, {{0, 5, 0}, {4, 5, 0}});
    const auto line = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::line, {{0, 0, 0}, {0, 1, 0}});
    const auto result = closest_pair(segment, line);
    check_scalar(result.distance2, Real{0});
    check_point<Real, Dims>(result.point0, {Real{0}, Real{5}, Real{0}});
    check_point<Real, Dims>(result.point1, {Real{0}, Real{5}, Real{0}});
    check_distance2_matches(segment, line, result.distance2);
  }
  {
    INFO("test_ray_ray_diverging");
    const auto a = geometric_primitive<Real, Dims>(tf::cpp::primitive_kind::ray,
                                                   {{0, 0, 0}, {1, 0, 0}});
    const auto b = geometric_primitive<Real, Dims>(tf::cpp::primitive_kind::ray,
                                                   {{0, 4, 0}, {1, 0, 0}});
    const auto result = closest_pair(a, b);
    check_scalar(result.distance2, Real{16});
    check_scalar(result.point0[0], result.point1[0]);
    check_distance2_matches(a, b, result.distance2);
  }
  {
    INFO("test_ray_line_perpendicular");
    const auto ray = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::ray, {{3, 0, 0}, {0, 1, 0}});
    const auto line = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::line, {{0, 0, 0}, {1, 0, 0}});
    const auto result = closest_pair(ray, line);
    check_scalar(result.distance2, Real{0});
    check_point<Real, Dims>(result.point0, {Real{3}, Real{0}, Real{0}});
    check_point<Real, Dims>(result.point1, {Real{3}, Real{0}, Real{0}});
    check_distance2_matches(ray, line, result.distance2);
  }
  {
    INFO("test_line_line_parallel");
    const auto a = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::line, {{0, 0, 0}, {1, 0, 0}});
    const auto b = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::line, {{0, 5, 0}, {1, 0, 0}});
    const auto result = closest_pair(a, b);
    check_scalar(result.distance2, Real{25});
    check_scalar(result.point0[0], result.point1[0]);
    check_distance2_matches(a, b, result.distance2);
  }
}

template <typename Real> auto python_line_line_skew_parity() -> void {
  INFO("test_line_line_skew_3d["
       << (std::is_same<Real, float>::value ? "float32" : "float64") << "]");
  const auto a = geometric_primitive<Real, 3>(tf::cpp::primitive_kind::line,
                                              {{0, 0, 0}, {1, 0, 0}});
  const auto b = geometric_primitive<Real, 3>(tf::cpp::primitive_kind::line,
                                              {{0, 0, 4}, {0, 1, 0}});
  const auto result = closest_pair(a, b);
  check_scalar(result.distance2, Real{16});
  check_point<Real, 3>(result.point0, {Real{0}, Real{0}, Real{0}});
  check_point<Real, 3>(result.point1, {Real{0}, Real{0}, Real{4}});
  check_distance2_matches(a, b, result.distance2);
}

} // namespace

TEST_CASE("Python closest pair polygon basics and compile-time dimension "
          "mismatch have native parity",
          "[cpp][spatial][python-parity][closest-pair]") {
  python_polygon_basic_parity();
}

TEST_CASE("Python closest pair point segment ray and line metrics have native "
          "parity",
          "[cpp][spatial][python-parity][closest-pair]") {
  python_metric_primitive_parity<float, 2>();
  python_metric_primitive_parity<float, 3>();
  python_metric_primitive_parity<double, 2>();
  python_metric_primitive_parity<double, 3>();
  python_line_line_skew_parity<float>();
  python_line_line_skew_parity<double>();
}

namespace {

template <typename Real, std::size_t Dims>
auto python_segment_polygon_parity() -> void {
  INFO("test_segment_polygon_separated["
       << (std::is_same<Real, float>::value ? "float32" : "float64")
       << ", dims=" << Dims << "]");
  const auto polygon = geometric_primitive<Real, Dims>(
      tf::cpp::primitive_kind::polygon,
      {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
  const auto segment = geometric_primitive<Real, Dims>(
      tf::cpp::primitive_kind::segment, {{3, Real{0.5}, 0}, {5, Real{0.5}, 0}});
  const auto result = closest_pair(segment, polygon);
  check_scalar(result.distance2, Real{4});
  check_point<Real, Dims>(result.point0, {Real{3}, Real{0.5}, Real{0}});
  check_point<Real, Dims>(result.point1, {Real{1}, Real{0.5}, Real{0}});
  check_distance2_matches(segment, polygon, result.distance2);
}

template <typename Real> auto python_polygon_plane_and_swap_parity() -> void {
  INFO("dtype=" << (std::is_same<Real, float>::value ? "float32" : "float64"));
  {
    INFO("test_polygon_ray_hitting_3d");
    const auto polygon =
        geometric_primitive<Real, 3>(tf::cpp::primitive_kind::polygon,
                                     {{0, 0, 0}, {1, 0, 0}, {Real{0.5}, 1, 0}});
    const auto ray =
        geometric_primitive<Real, 3>(tf::cpp::primitive_kind::ray,
                                     {{Real{0.25}, Real{0.25}, 5}, {0, 0, -1}});
    const auto result = closest_pair(polygon, ray);
    check_scalar(result.distance2, Real{0});
    check_distance2_matches(polygon, ray, result.distance2);
  }
  {
    INFO("test_polygon_line_parallel_3d");
    const auto polygon =
        geometric_primitive<Real, 3>(tf::cpp::primitive_kind::polygon,
                                     {{0, 0, 0}, {1, 0, 0}, {Real{0.5}, 1, 0}});
    const auto line = geometric_primitive<Real, 3>(
        tf::cpp::primitive_kind::line, {{0, Real{0.5}, 3}, {1, 0, 0}});
    const auto result = closest_pair(polygon, line);
    check_scalar(result.distance2, Real{9});
    check_scalar(result.point0[2], Real{0});
    check_scalar(result.point1[2], Real{3});
    check_distance2_matches(polygon, line, result.distance2);
  }
  {
    INFO("test_polygon_plane_parallel");
    const auto polygon =
        geometric_primitive<Real, 3>(tf::cpp::primitive_kind::polygon,
                                     {{0, 0, 0}, {1, 0, 0}, {Real{0.5}, 1, 0}});
    const auto plane = plane_d<Real>(0, 0, 1, -5);
    const auto result = closest_pair(polygon, plane);
    check_scalar(result.distance2, Real{25});
    check_scalar(result.point0[2], Real{0});
    check_scalar(result.point1[2], Real{5});
    check_distance2_matches(polygon, plane, result.distance2);
  }
  {
    INFO("test_plane_plane_parallel");
    const auto a = plane_d<Real>(0, 0, 1, 0);
    const auto b = plane_d<Real>(0, 0, 1, -7);
    const auto result = closest_pair(a, b);
    check_scalar(result.distance2, Real{49});
    check_scalar(result.point0[2], Real{0});
    check_scalar(result.point1[2], Real{7});
    check_distance2_matches(a, b, result.distance2);
  }
  {
    INFO("test_plane_plane_intersecting");
    const auto a = plane_d<Real>(0, 0, 1, 0);
    const auto b = plane_d<Real>(1, 0, 0, 0);
    const auto result = closest_pair(a, b);
    check_scalar(result.distance2, Real{0});
    check_distance2_matches(a, b, result.distance2);
  }
  {
    INFO("test_swap_symmetry_point_segment");
    const auto point = point_d<Real, 3>(2, 3, 0);
    const auto segment = geometric_primitive<Real, 3>(
        tf::cpp::primitive_kind::segment, {{0, 0, 0}, {4, 0, 0}});
    const auto forward = closest_pair(point, segment);
    const auto reverse = closest_pair(segment, point);
    check_scalar(forward.distance2, reverse.distance2);
    check_points_equal<Real, 3>(forward.point0, reverse.point1);
    check_points_equal<Real, 3>(forward.point1, reverse.point0);
  }
  {
    INFO("test_swap_symmetry_segment_polygon");
    const auto polygon = geometric_primitive<Real, 3>(
        tf::cpp::primitive_kind::polygon,
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
    const auto segment =
        geometric_primitive<Real, 3>(tf::cpp::primitive_kind::segment,
                                     {{3, Real{0.5}, 0}, {5, Real{0.5}, 0}});
    const auto forward = closest_pair(segment, polygon);
    const auto reverse = closest_pair(polygon, segment);
    check_scalar(forward.distance2, reverse.distance2);
    check_points_equal<Real, 3>(forward.point0, reverse.point1);
    check_points_equal<Real, 3>(forward.point1, reverse.point0);
  }
}

} // namespace

TEST_CASE("Python closest pair polygon plane and swap symmetry have native "
          "parity",
          "[cpp][spatial][python-parity][closest-pair]") {
  python_segment_polygon_parity<float, 2>();
  python_segment_polygon_parity<float, 3>();
  python_segment_polygon_parity<double, 2>();
  python_segment_polygon_parity<double, 3>();
  python_polygon_plane_and_swap_parity<float>();
  python_polygon_plane_and_swap_parity<double>();
}

namespace {

template <typename Real, std::size_t Dims>
auto box_d(const std::array<Real, 3> &minimum,
           const std::array<Real, 3> &maximum)
    -> tf::cpp::primitive<Real, Dims> {
  return geometric_primitive<Real, Dims>(tf::cpp::primitive_kind::aabb,
                                         {minimum, maximum});
}

template <typename Real, std::size_t Dims>
auto unit_box_d() -> tf::cpp::primitive<Real, Dims> {
  return box_d<Real, Dims>({Real{0}, Real{0}, Real{0}},
                           {Real{1}, Real{1}, Real{1}});
}

template <typename Real, std::size_t Dims>
auto python_aabb_family_parity() -> void {
  INFO("dtype=" << (std::is_same<Real, float>::value ? "float32" : "float64")
                << ", dims=" << Dims);
  {
    INFO("test_aabb_point_outside");
    const auto box = unit_box_d<Real, Dims>();
    const auto query = point_d<Real, Dims>(2, Real{0.5}, 0);
    const auto result = closest_pair(box, query);
    check_scalar(result.distance2, Real{1});
    check_point<Real, Dims>(result.point0, {Real{1}, Real{0.5}, Real{0}});
    check_point<Real, Dims>(result.point1, {Real{2}, Real{0.5}, Real{0}});
    check_distance2_matches(box, query, result.distance2);
  }
  {
    INFO("test_aabb_point_inside");
    const auto box = unit_box_d<Real, Dims>();
    const auto query = point_d<Real, Dims>(Real{0.5}, Real{0.5}, Real{0.5});
    check_scalar(closest_pair(box, query).distance2, Real{0});
  }
  {
    INFO("test_aabb_point_corner");
    const auto box = unit_box_d<Real, Dims>();
    const auto query = point_d<Real, Dims>(2, 2, 2);
    const auto result = closest_pair(box, query);
    check_scalar(result.distance2, static_cast<Real>(Dims));
    check_point<Real, Dims>(result.point0, {Real{1}, Real{1}, Real{1}});
    check_distance2_matches(box, query, result.distance2);
  }
  {
    INFO("test_aabb_segment_outside");
    const auto box = unit_box_d<Real, Dims>();
    const auto segment = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment, {{2, 0, 0}, {2, 1, 0}});
    const auto result = closest_pair(box, segment);
    check_scalar(result.distance2, Real{1});
    check_scalar(result.point0[0], Real{1});
    check_scalar(result.point1[0], Real{2});
    check_distance2_matches(box, segment, result.distance2);
  }
  {
    INFO("test_aabb_segment_crossing");
    const auto box = unit_box_d<Real, Dims>();
    const auto segment = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::segment,
        {{-1, Real{0.5}, 0}, {2, Real{0.5}, 0}});
    check_scalar(closest_pair(box, segment).distance2, Real{0});
  }
  {
    INFO("test_aabb_ray_hitting");
    const auto box = unit_box_d<Real, Dims>();
    const auto ray = [&] {
      if constexpr (Dims == 2)
        return geometric_primitive<Real, Dims>(
            tf::cpp::primitive_kind::ray,
            {{Real{0.5}, Real{5}, 0}, {Real{0}, Real{-1}, 0}});
      else
        return geometric_primitive<Real, Dims>(
            tf::cpp::primitive_kind::ray,
            {{Real{0.5}, Real{0.5}, Real{5}}, {Real{0}, Real{0}, Real{-1}}});
    }();
    check_scalar(closest_pair(box, ray).distance2, Real{0});
  }
  {
    INFO("test_aabb_ray_missing");
    const auto box = unit_box_d<Real, Dims>();
    const auto ray = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::ray, {{2, Real{0.5}, Real{0.5}}, {1, 0, 0}});
    const auto result = closest_pair(box, ray);
    check_scalar(result.distance2, Real{1});
    check_scalar(result.point0[0], Real{1});
    check_scalar(result.point1[0], Real{2});
    check_distance2_matches(box, ray, result.distance2);
  }
  {
    INFO("test_aabb_line_intersecting");
    const auto box = unit_box_d<Real, Dims>();
    const auto line = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::line, {{-5, Real{0.5}, Real{0.5}}, {1, 0, 0}});
    check_scalar(closest_pair(box, line).distance2, Real{0});
  }
  {
    INFO("test_aabb_line_parallel");
    const auto box = unit_box_d<Real, Dims>();
    const auto line = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::line, {{3, 0, 0}, {0, 1, 0}});
    const auto result = closest_pair(box, line);
    check_scalar(result.distance2, Real{4});
    check_scalar(result.point0[0], Real{1});
    check_scalar(result.point1[0], Real{3});
    check_distance2_matches(box, line, result.distance2);
  }
  {
    INFO("test_aabb_polygon_separated");
    const auto box = unit_box_d<Real, Dims>();
    const auto polygon = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::polygon,
        {{3, 0, 0}, {4, 0, 0}, {Real{3.5}, 1, 0}});
    const auto result = closest_pair(box, polygon);
    check_scalar(result.distance2, Real{4});
    check_scalar(result.point0[0], Real{1});
    check_scalar(result.point1[0], Real{3});
    check_distance2_matches(box, polygon, result.distance2);
  }
  {
    INFO("test_aabb_polygon_overlapping");
    const auto box = unit_box_d<Real, Dims>();
    const auto polygon = geometric_primitive<Real, Dims>(
        tf::cpp::primitive_kind::polygon,
        {{Real{0.5}, Real{0.5}, 0}, {2, Real{0.5}, 0}, {Real{0.5}, 2, 0}});
    check_scalar(closest_pair(box, polygon).distance2, Real{0});
  }
  {
    INFO("test_aabb_aabb_separated");
    const auto a = unit_box_d<Real, Dims>();
    const auto b = box_d<Real, Dims>({Real{3}, Real{0}, Real{0}},
                                     {Real{4}, Real{1}, Real{1}});
    const auto result = closest_pair(a, b);
    check_scalar(result.distance2, Real{4});
    check_scalar(result.point0[0], Real{1});
    check_scalar(result.point1[0], Real{3});
    check_distance2_matches(a, b, result.distance2);
  }
  {
    INFO("test_aabb_aabb_overlapping");
    const auto a = unit_box_d<Real, Dims>();
    const auto b = box_d<Real, Dims>({Real{0.5}, Real{0.5}, Real{0.5}},
                                     {Real{1.5}, Real{1.5}, Real{1.5}});
    check_scalar(closest_pair(a, b).distance2, Real{0});
  }
  {
    INFO("test_aabb_aabb_touching");
    const auto a = unit_box_d<Real, Dims>();
    const auto b = box_d<Real, Dims>({Real{1}, Real{0}, Real{0}},
                                     {Real{2}, Real{1}, Real{1}});
    check_scalar(closest_pair(a, b).distance2, Real{0});
  }
  {
    INFO("test_aabb_aabb_contained");
    const auto a = box_d<Real, Dims>({Real{0}, Real{0}, Real{0}},
                                     {Real{4}, Real{4}, Real{4}});
    const auto b = box_d<Real, Dims>({Real{1}, Real{1}, Real{1}},
                                     {Real{2}, Real{2}, Real{2}});
    check_scalar(closest_pair(a, b).distance2, Real{0});
  }
}

template <typename Real> auto python_aabb_plane_and_swap_parity() -> void {
  INFO("dtype=" << (std::is_same<Real, float>::value ? "float32" : "float64"));
  {
    INFO("test_aabb_plane_intersecting");
    const auto box = unit_box_d<Real, 3>();
    const auto plane = plane_d<Real>(0, 0, 1, Real{-0.5});
    check_scalar(closest_pair(box, plane).distance2, Real{0});
  }
  {
    INFO("test_aabb_plane_separated");
    const auto box = unit_box_d<Real, 3>();
    const auto plane = plane_d<Real>(0, 0, 1, -4);
    const auto result = closest_pair(box, plane);
    check_scalar(result.distance2, Real{9});
    check_scalar(result.point0[2], Real{1});
    check_scalar(result.point1[2], Real{4});
    check_distance2_matches(box, plane, result.distance2);
  }
  {
    INFO("test_swap_symmetry_aabb_point");
    const auto box = unit_box_d<Real, 3>();
    const auto point = point_d<Real, 3>(2, Real{0.5}, Real{0.5});
    const auto forward = closest_pair(box, point);
    const auto reverse = closest_pair(point, box);
    check_scalar(forward.distance2, reverse.distance2);
    check_points_equal<Real, 3>(forward.point0, reverse.point1);
    check_points_equal<Real, 3>(forward.point1, reverse.point0);
  }
  {
    INFO("test_swap_symmetry_aabb_segment");
    const auto box = unit_box_d<Real, 3>();
    const auto segment = geometric_primitive<Real, 3>(
        tf::cpp::primitive_kind::segment,
        {{3, Real{0.5}, Real{0.5}}, {5, Real{0.5}, Real{0.5}}});
    const auto forward = closest_pair(box, segment);
    const auto reverse = closest_pair(segment, box);
    check_scalar(forward.distance2, reverse.distance2);
    check_points_equal<Real, 3>(forward.point0, reverse.point1);
    check_points_equal<Real, 3>(forward.point1, reverse.point0);
  }
  {
    INFO("test_swap_symmetry_aabb_aabb");
    const auto a = unit_box_d<Real, 3>();
    const auto b = box_d<Real, 3>({Real{3}, Real{0}, Real{0}},
                                  {Real{4}, Real{1}, Real{1}});
    const auto forward = closest_pair(a, b);
    const auto reverse = closest_pair(b, a);
    check_scalar(forward.distance2, reverse.distance2);
    check_points_equal<Real, 3>(forward.point0, reverse.point1);
    check_points_equal<Real, 3>(forward.point1, reverse.point0);
  }
  {
    INFO("test_swap_symmetry_aabb_polygon");
    const auto box = unit_box_d<Real, 3>();
    const auto polygon =
        geometric_primitive<Real, 3>(tf::cpp::primitive_kind::polygon,
                                     {{3, 0, 0}, {4, 0, 0}, {Real{3.5}, 1, 0}});
    const auto forward = closest_pair(box, polygon);
    const auto reverse = closest_pair(polygon, box);
    check_scalar(forward.distance2, reverse.distance2);
    check_points_equal<Real, 3>(forward.point0, reverse.point1);
    check_points_equal<Real, 3>(forward.point1, reverse.point0);
  }
}

} // namespace

TEST_CASE("Python closest pair AABB families and swaps have native parity",
          "[cpp][spatial][python-parity][closest-pair]") {
  python_aabb_family_parity<float, 2>();
  python_aabb_family_parity<float, 3>();
  python_aabb_family_parity<double, 2>();
  python_aabb_family_parity<double, 3>();
  python_aabb_plane_and_swap_parity<float>();
  python_aabb_plane_and_swap_parity<double>();
}
