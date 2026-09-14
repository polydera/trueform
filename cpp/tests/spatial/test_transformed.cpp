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
#include "trueform/cpp/spatial/async/transformed.hpp"
#include "trueform/cpp/spatial/transformed.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <future>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Real>
auto make_array(std::initializer_list<Real> values,
                tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<Real> {
  tf::buffer<Real> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<Real>::from_buffer(std::move(buffer),
                                              std::move(shape));
}

template <typename Real>
auto make_array(const std::vector<Real> &values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<Real> {
  tf::buffer<Real> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<Real>::from_buffer(std::move(buffer),
                                              std::move(shape));
}

template <typename Real>
auto make_empty(tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<Real> {
  tf::buffer<Real> buffer;
  buffer.allocate(0);
  return tf::cpp::nd_array<Real>::from_buffer(std::move(buffer),
                                              std::move(shape));
}

template <typename Real, std::size_t Dims>
auto primitive(tf::cpp::primitive_kind kind, std::initializer_list<Real> values,
               tf::small_vector<int, 3> shape)
    -> tf::cpp::primitive<Real, Dims> {
  return tf::cpp::primitive<Real, Dims>(
      kind, make_array<Real>(values, std::move(shape)));
}

template <typename Real, std::size_t Dims>
auto identity_matrix() -> tf::cpp::nd_array<Real> {
  constexpr auto side = Dims + 1;
  std::vector<Real> values(side * side, Real{0});
  for (std::size_t axis = 0; axis < side; ++axis)
    values[axis * side + axis] = Real{1};
  return make_array(values, {static_cast<int>(side), static_cast<int>(side)});
}

template <typename Real>
auto rotation_translation_2d(Real tx = Real{0}, Real ty = Real{0})
    -> tf::cpp::nd_array<Real> {
  return make_array<Real>({0, -1, tx, 1, 0, ty, 0, 0, 1}, {3, 3});
}

template <typename Real>
auto rotation_translation_z_3d(Real tx = Real{0}, Real ty = Real{0},
                               Real tz = Real{0}) -> tf::cpp::nd_array<Real> {
  return make_array<Real>({0, -1, 0, tx, 1, 0, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1},
                          {4, 4});
}

template <typename Real>
auto translation_3d(Real tx, Real ty, Real tz) -> tf::cpp::nd_array<Real> {
  return make_array<Real>({1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1},
                          {4, 4});
}

template <typename Real>
auto rotation_translation_45_2d(Real tx, Real ty) -> tf::cpp::nd_array<Real> {
  const auto cosine = std::sqrt(Real{0.5});
  return make_array<Real>({cosine, -cosine, tx, cosine, cosine, ty, 0, 0, 1},
                          {3, 3});
}

template <typename Real>
auto rotation_translation_45_z_3d(Real tx, Real ty, Real tz)
    -> tf::cpp::nd_array<Real> {
  const auto cosine = std::sqrt(Real{0.5});
  return make_array<Real>(
      {cosine, -cosine, 0, tx, cosine, cosine, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1},
      {4, 4});
}

template <typename Real> constexpr auto tolerance() -> double {
  return std::is_same<Real, float>::value ? 1e-5 : 1e-11;
}

template <typename Real>
auto require_shape(const tf::cpp::nd_array<Real> &array,
                   std::initializer_list<int> expected) -> void {
  REQUIRE(array.ndim() == static_cast<int>(expected.size()));
  int axis = 0;
  for (const auto extent : expected)
    CHECK(array.shape_at(axis++) == extent);
}

template <typename Real>
auto check_values(const tf::cpp::nd_array<Real> &array,
                  std::initializer_list<double> expected,
                  double margin = tolerance<Real>()) -> void {
  REQUIRE(array.length() == expected.size());
  std::size_t index = 0;
  for (const auto value : expected)
    CHECK(array[index++] == Catch::Approx(value).margin(margin));
}

template <typename Real>
auto check_exactly_unchanged(const tf::cpp::nd_array<Real> &actual,
                             const tf::cpp::nd_array<Real> &snapshot) -> void {
  REQUIRE(actual.length() == snapshot.length());
  REQUIRE(actual.ndim() == snapshot.ndim());
  for (int axis = 0; axis < actual.ndim(); ++axis)
    REQUIRE(actual.shape_at(axis) == snapshot.shape_at(axis));
  for (std::size_t index = 0; index < actual.length(); ++index)
    CHECK(actual[index] == snapshot[index]);
}

template <typename Real, std::size_t Dims>
auto check_batch_metadata(const tf::cpp::primitive<Real, Dims> &value,
                          tf::cpp::primitive_kind kind, int count,
                          std::initializer_list<int> shape) -> void {
  CHECK(value.kind() == kind);
  CHECK(value.cardinality() == tf::cpp::primitive_cardinality::batch);
  CHECK(value.is_batch());
  CHECK(value.count() == count);
  require_shape(value.data(), shape);
}

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

template <typename Real, std::size_t Dims>
auto translated_point() -> tf::cpp::primitive<Real, Dims> {
  if constexpr (Dims == 2) {
    const auto source =
        primitive<Real, Dims>(tf::cpp::primitive_kind::point, {1, 0}, {2});
    const auto matrix = rotation_translation_2d<Real>(2, 3);
    return tf::cpp::transformed(source, matrix);
  } else {
    const auto source =
        primitive<Real, Dims>(tf::cpp::primitive_kind::point, {1, 0, 0}, {3});
    const auto matrix = rotation_translation_z_3d<Real>(2, 3, 4);
    return tf::cpp::transformed(source, matrix);
  }
}

template <typename Real, std::size_t Dims>
auto translated_point_future() -> std::future<tf::cpp::primitive<Real, Dims>> {
  if constexpr (Dims == 2) {
    const auto source =
        primitive<Real, Dims>(tf::cpp::primitive_kind::point, {1, 0}, {2});
    const auto matrix = rotation_translation_2d<Real>(2, 3);
    return tf::cpp::async::transformed(source, matrix);
  } else {
    const auto source =
        primitive<Real, Dims>(tf::cpp::primitive_kind::point, {1, 0, 0}, {3});
    const auto matrix = rotation_translation_z_3d<Real>(2, 3, 4);
    return tf::cpp::async::transformed(source, matrix);
  }
}

template <typename Real, std::size_t Dims>
auto translated_point_custom(const counting_resolver &resolver)
    -> std::future<tf::cpp::primitive<Real, Dims>> {
  if constexpr (Dims == 2) {
    const auto source =
        primitive<Real, Dims>(tf::cpp::primitive_kind::point, {1, 0}, {2});
    const auto matrix = rotation_translation_2d<Real>(2, 3);
    return tf::cpp::async::transformed(resolver, source, matrix);
  } else {
    const auto source =
        primitive<Real, Dims>(tf::cpp::primitive_kind::point, {1, 0, 0}, {3});
    const auto matrix = rotation_translation_z_3d<Real>(2, 3, 4);
    return tf::cpp::async::transformed(resolver, source, matrix);
  }
}

template <typename Real, std::size_t Dims>
auto empty_primitive(tf::cpp::primitive_kind kind)
    -> tf::cpp::primitive<Real, Dims> {
  const auto dimension = static_cast<int>(Dims);
  switch (kind) {
  case tf::cpp::primitive_kind::point:
  case tf::cpp::primitive_kind::vector:
    return tf::cpp::primitive<Real, Dims>(kind,
                                          make_empty<Real>({0, dimension}));
  case tf::cpp::primitive_kind::segment:
  case tf::cpp::primitive_kind::ray:
  case tf::cpp::primitive_kind::line:
  case tf::cpp::primitive_kind::aabb:
    return tf::cpp::primitive<Real, Dims>(kind,
                                          make_empty<Real>({0, 2, dimension}));
  case tf::cpp::primitive_kind::triangle:
    return tf::cpp::primitive<Real, Dims>(kind,
                                          make_empty<Real>({0, 3, dimension}));
  case tf::cpp::primitive_kind::plane:
    if constexpr (Dims == 3)
      return tf::cpp::primitive<Real, Dims>(kind, make_empty<Real>({0, 4}));
    else
      throw std::logic_error("2D planes are not runtime primitives");
  case tf::cpp::primitive_kind::polygon:
    return tf::cpp::primitive<Real, Dims>(kind,
                                          make_empty<Real>({0, 4, dimension}));
  }
  throw std::logic_error("unknown primitive kind");
}

template <typename Real, std::size_t Dims> auto check_empty_batches() -> void {
  constexpr std::array kinds{
      tf::cpp::primitive_kind::point,   tf::cpp::primitive_kind::vector,
      tf::cpp::primitive_kind::segment, tf::cpp::primitive_kind::triangle,
      tf::cpp::primitive_kind::ray,     tf::cpp::primitive_kind::line,
      tf::cpp::primitive_kind::aabb,    tf::cpp::primitive_kind::polygon};
  const auto matrix = identity_matrix<Real, Dims>();
  for (const auto kind : kinds) {
    INFO("empty primitive kind " << static_cast<int>(kind));
    const auto source = empty_primitive<Real, Dims>(kind);
    const auto result = tf::cpp::transformed(source, matrix);
    CHECK(result.kind() == kind);
    CHECK(result.cardinality() == tf::cpp::primitive_cardinality::batch);
    CHECK(result.is_batch());
    CHECK(result.count() == 0);
    CHECK(result.data().empty());
    CHECK(result.data().raw_shape() == source.data().raw_shape());
  }
  if constexpr (Dims == 3) {
    const auto source =
        empty_primitive<Real, Dims>(tf::cpp::primitive_kind::plane);
    const auto result = tf::cpp::transformed(source, matrix);
    CHECK(result.kind() == tf::cpp::primitive_kind::plane);
    CHECK(result.is_batch());
    CHECK(result.count() == 0);
    CHECK(result.data().empty());
    CHECK(result.data().raw_shape() == source.data().raw_shape());
  }
}

template <typename Real, std::size_t Dims>
auto check_detached_output() -> void {
  auto source = [&] {
    if constexpr (Dims == 2)
      return primitive<Real, Dims>(tf::cpp::primitive_kind::point, {1, 0, 0, 1},
                                   {2, 2});
    else
      return primitive<Real, Dims>(tf::cpp::primitive_kind::point,
                                   {1, 0, 0, 0, 1, 0}, {2, 3});
  }();
  auto matrix = [&] {
    if constexpr (Dims == 2)
      return rotation_translation_2d<Real>(2, 3);
    else
      return rotation_translation_z_3d<Real>(2, 3, 4);
  }();
  const auto source_snapshot = source.data().deep_copy();
  const auto matrix_snapshot = matrix.deep_copy();

  auto result = tf::cpp::transformed(source, matrix);
  check_exactly_unchanged(source.data(), source_snapshot);
  check_exactly_unchanged(matrix, matrix_snapshot);
  REQUIRE(result.data().raw_data() != source.data().raw_data());

  auto result_data = result.data();
  result_data[0] = Real{77};
  CHECK(source.data()[0] == source_snapshot[0]);
  CHECK(matrix[0] == matrix_snapshot[0]);

  auto source_data = source.data();
  source_data[0] = Real{99};
  CHECK(result.data()[0] == Real{77});
}

template <typename Real, std::size_t Dims>
auto check_async_ownership() -> void {
  const auto sync = translated_point<Real, Dims>();
  check_values(sync.data(), Dims == 2 ? std::initializer_list<double>{2, 4}
                                      : std::initializer_list<double>{2, 4, 4});

  auto future = translated_point_future<Real, Dims>();
  STATIC_REQUIRE(std::is_same_v<decltype(future),
                                std::future<tf::cpp::primitive<Real, Dims>>>);
  const auto from_future = future.get();
  check_values(from_future.data(),
               Dims == 2 ? std::initializer_list<double>{2, 4}
                         : std::initializer_list<double>{2, 4, 4});

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom =
      translated_point_custom<Real, Dims>(counting_resolver{submissions});
  STATIC_REQUIRE(std::is_same_v<decltype(custom),
                                std::future<tf::cpp::primitive<Real, Dims>>>);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  const auto from_custom = custom.get();
  check_values(from_custom.data(),
               Dims == 2 ? std::initializer_list<double>{2, 4}
                         : std::initializer_list<double>{2, 4, 4});
}

} // namespace

TEMPLATE_TEST_CASE("Python point transformations have 2D and 3D parity",
                   "[cpp][spatial][transformed][python-parity][point]", float,
                   double) {
  const auto point2 =
      primitive<TestType, 2>(tf::cpp::primitive_kind::point, {1, 0}, {2});
  const auto rotated2 =
      tf::cpp::transformed(point2, rotation_translation_2d<TestType>());
  CHECK_FALSE(rotated2.is_batch());
  CHECK(rotated2.kind() == tf::cpp::primitive_kind::point);
  require_shape(rotated2.data(), {2});
  check_values(rotated2.data(), {0, 1});

  const auto moved2 =
      tf::cpp::transformed(point2, rotation_translation_2d<TestType>(2, 3));
  check_values(moved2.data(), {2, 4});

  const auto point3 =
      primitive<TestType, 3>(tf::cpp::primitive_kind::point, {1, 0, 0}, {3});
  const auto rotated3 =
      tf::cpp::transformed(point3, rotation_translation_z_3d<TestType>());
  CHECK_FALSE(rotated3.is_batch());
  CHECK(rotated3.kind() == tf::cpp::primitive_kind::point);
  require_shape(rotated3.data(), {3});
  check_values(rotated3.data(), {0, 1, 0});

  const auto moved3 = tf::cpp::transformed(
      point3, rotation_translation_z_3d<TestType>(2, 3, 4));
  check_values(moved3.data(), {2, 4, 4});
}

TEMPLATE_TEST_CASE(
    "Python segment and polygon transformations preserve vertices and shapes",
    "[cpp][spatial][transformed][python-parity][vertices]", float, double) {
  const auto segment2 = primitive<TestType, 2>(tf::cpp::primitive_kind::segment,
                                               {1, 0, 0, 1}, {2, 2});
  const auto transformed_segment2 =
      tf::cpp::transformed(segment2, rotation_translation_2d<TestType>());
  require_shape(transformed_segment2.data(), {2, 2});
  check_values(transformed_segment2.data(), {0, 1, -1, 0});

  const auto segment3 = primitive<TestType, 3>(tf::cpp::primitive_kind::segment,
                                               {1, 0, 0, 0, 1, 0}, {2, 3});
  const auto transformed_segment3 = tf::cpp::transformed(
      segment3, rotation_translation_z_3d<TestType>(2, 3, 4));
  require_shape(transformed_segment3.data(), {2, 3});
  check_values(transformed_segment3.data(), {2, 4, 4, 1, 3, 4});

  const auto polygon2 = primitive<TestType, 2>(tf::cpp::primitive_kind::polygon,
                                               {1, 0, 0, 1, -1, 0}, {3, 2});
  const auto transformed_polygon2 =
      tf::cpp::transformed(polygon2, rotation_translation_2d<TestType>());
  CHECK(transformed_polygon2.polygon_vertex_count() == 3);
  require_shape(transformed_polygon2.data(), {3, 2});
  check_values(transformed_polygon2.data(), {0, 1, -1, 0, 0, -1});

  const auto polygon3 = primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3});
  const auto transformed_polygon3 =
      tf::cpp::transformed(polygon3, rotation_translation_z_3d<TestType>());
  CHECK(transformed_polygon3.polygon_vertex_count() == 3);
  require_shape(transformed_polygon3.data(), {3, 3});
  check_values(transformed_polygon3.data(), {0, 1, 0, -1, 0, 0, 0, 0, 1});
}

TEMPLATE_TEST_CASE(
    "vector and triangle transformations close the runtime primitive enum",
    "[cpp][spatial][transformed][enum][vector][triangle]", float, double) {
  const auto vector2 =
      primitive<TestType, 2>(tf::cpp::primitive_kind::vector, {1, 0}, {2});
  const auto transformed_vector2 =
      tf::cpp::transformed(vector2, rotation_translation_2d<TestType>(20, 30));
  CHECK(transformed_vector2.kind() == tf::cpp::primitive_kind::vector);
  check_values(transformed_vector2.data(), {0, 1});

  const auto vector3 =
      primitive<TestType, 3>(tf::cpp::primitive_kind::vector, {1, 0, 2}, {3});
  const auto transformed_vector3 = tf::cpp::transformed(
      vector3, rotation_translation_z_3d<TestType>(20, 30, 40));
  CHECK(transformed_vector3.kind() == tf::cpp::primitive_kind::vector);
  check_values(transformed_vector3.data(), {0, 1, 2});

  const auto triangle2 = primitive<TestType, 2>(
      tf::cpp::primitive_kind::triangle, {1, 0, 0, 1, 0, 0}, {3, 2});
  const auto transformed_triangle2 =
      tf::cpp::transformed(triangle2, rotation_translation_2d<TestType>(2, 3));
  CHECK(transformed_triangle2.kind() == tf::cpp::primitive_kind::triangle);
  require_shape(transformed_triangle2.data(), {3, 2});
  check_values(transformed_triangle2.data(), {2, 4, 1, 3, 2, 3});

  const auto triangle3 = primitive<TestType, 3>(
      tf::cpp::primitive_kind::triangle, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3});
  const auto transformed_triangle3 = tf::cpp::transformed(
      triangle3, rotation_translation_z_3d<TestType>(2, 3, 4));
  CHECK(transformed_triangle3.kind() == tf::cpp::primitive_kind::triangle);
  require_shape(transformed_triangle3.data(), {3, 3});
  check_values(transformed_triangle3.data(), {2, 4, 4, 1, 3, 4, 2, 3, 5});
}

TEMPLATE_TEST_CASE(
    "Python AABB transformations preserve identity ordering and exact bounds",
    "[cpp][spatial][transformed][python-parity][aabb]", float, double) {
  const auto cosine = std::sqrt(0.5);
  const auto box2 = primitive<TestType, 2>(tf::cpp::primitive_kind::aabb,
                                           {0, 0, 1, 1}, {2, 2});
  const auto transformed_box2 =
      tf::cpp::transformed(box2, rotation_translation_45_2d<TestType>(2, 3));
  require_shape(transformed_box2.data(), {2, 2});
  check_values(transformed_box2.data(),
               {2 - cosine, 3, 2 + cosine, 3 + 2 * cosine}, 1e-5);
  CHECK(transformed_box2.data()[0] <= transformed_box2.data()[2]);
  CHECK(transformed_box2.data()[1] <= transformed_box2.data()[3]);

  const auto box3 = primitive<TestType, 3>(tf::cpp::primitive_kind::aabb,
                                           {0, 0, 0, 1, 1, 1}, {2, 3});
  const auto transformed_box3 = tf::cpp::transformed(
      box3, rotation_translation_45_z_3d<TestType>(2, 3, 4));
  require_shape(transformed_box3.data(), {2, 3});
  check_values(transformed_box3.data(),
               {2 - cosine, 3, 4, 2 + cosine, 3 + 2 * cosine, 5}, 1e-5);
  for (std::size_t axis = 0; axis < 3; ++axis)
    CHECK(transformed_box3.data()[axis] <= transformed_box3.data()[axis + 3]);

  const auto identity_box = primitive<TestType, 3>(
      tf::cpp::primitive_kind::aabb, {0, 0, 0, 2, 3, 4}, {2, 3});
  const auto unchanged =
      tf::cpp::transformed(identity_box, identity_matrix<TestType, 3>());
  check_values(unchanged.data(), {0, 0, 0, 2, 3, 4});
}

TEMPLATE_TEST_CASE("Python ray and line transformations use affine origins and "
                   "unit directions",
                   "[cpp][spatial][transformed][python-parity][direction]",
                   float, double) {
  const auto ray2 = primitive<TestType, 2>(tf::cpp::primitive_kind::ray,
                                           {0, 0, 1, 0}, {2, 2});
  const auto transformed_ray2 =
      tf::cpp::transformed(ray2, rotation_translation_2d<TestType>());
  check_values(transformed_ray2.data(), {0, 0, 0, 1});
  CHECK(std::hypot(transformed_ray2.data()[2], transformed_ray2.data()[3]) ==
        Catch::Approx(1).margin(tolerance<TestType>()));

  const auto ray3 = primitive<TestType, 3>(tf::cpp::primitive_kind::ray,
                                           {0, 0, 0, 1, 0, 0}, {2, 3});
  const auto transformed_ray3 =
      tf::cpp::transformed(ray3, rotation_translation_z_3d<TestType>(2, 3, 4));
  check_values(transformed_ray3.data(), {2, 3, 4, 0, 1, 0});
  CHECK(std::hypot(transformed_ray3.data()[3], transformed_ray3.data()[4],
                   transformed_ray3.data()[5]) ==
        Catch::Approx(1).margin(tolerance<TestType>()));

  const auto line2 = primitive<TestType, 2>(tf::cpp::primitive_kind::line,
                                            {0, 0, 1, 0}, {2, 2});
  const auto transformed_line2 =
      tf::cpp::transformed(line2, rotation_translation_2d<TestType>(1, 2));
  check_values(transformed_line2.data(), {1, 2, 0, 1});
  CHECK(std::hypot(transformed_line2.data()[2], transformed_line2.data()[3]) ==
        Catch::Approx(1).margin(tolerance<TestType>()));

  const auto line3 = primitive<TestType, 3>(tf::cpp::primitive_kind::line,
                                            {1, 0, 0, 1, 0, 0}, {2, 3});
  const auto transformed_line3 =
      tf::cpp::transformed(line3, rotation_translation_z_3d<TestType>());
  check_values(transformed_line3.data(), {0, 1, 0, 0, 1, 0});
  CHECK(std::hypot(transformed_line3.data()[3], transformed_line3.data()[4],
                   transformed_line3.data()[5]) ==
        Catch::Approx(1).margin(tolerance<TestType>()));

  const auto nonunit_line = primitive<TestType, 2>(
      tf::cpp::primitive_kind::line, {1, 2, 2, 2}, {2, 2});
  const auto nonuniform =
      make_array<TestType>({2, 0, 3, 0, TestType{0.5}, 4, 0, 0, 1}, {3, 3});
  const auto normalized_line = tf::cpp::transformed(nonunit_line, nonuniform);
  const auto inverse_length = TestType{1} / std::sqrt(TestType{17});
  check_values(normalized_line.data(),
               {5, 5, 4 * inverse_length, inverse_length});
}

TEMPLATE_TEST_CASE(
    "3D plane transformations handle translation rotation and nonuniform scale",
    "[cpp][spatial][transformed][python-parity][plane]", float, double) {
  const auto plane =
      primitive<TestType, 3>(tf::cpp::primitive_kind::plane, {0, 0, 1, 0}, {4});
  const auto translated =
      tf::cpp::transformed(plane, translation_3d<TestType>(0, 0, 5));
  check_values(translated.data(), {0, 0, 1, -5});
  CHECK(std::hypot(translated.data()[0], translated.data()[1],
                   translated.data()[2]) ==
        Catch::Approx(1).margin(tolerance<TestType>()));

  const auto rotate_y = make_array<TestType>(
      {0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1}, {4, 4});
  const auto rotated = tf::cpp::transformed(plane, rotate_y);
  check_values(rotated.data(), {1, 0, 0, 0});

  const auto inverse_sqrt2 = TestType{1} / std::sqrt(TestType{2});
  const auto diagonal_plane = primitive<TestType, 3>(
      tf::cpp::primitive_kind::plane,
      {inverse_sqrt2, 0, inverse_sqrt2, -std::sqrt(TestType{2})}, {4});
  const auto nonuniform_scale = make_array<TestType>(
      {2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1}, {4, 4});
  const auto scaled = tf::cpp::transformed(diagonal_plane, nonuniform_scale);
  const auto sqrt5 = std::sqrt(5.0);
  check_values(scaled.data(), {2 / sqrt5, 0, 1 / sqrt5, -8 / sqrt5}, 1e-5);

  const auto singular = make_array<TestType>(
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, {4, 4});
  CHECK_THROWS_AS(tf::cpp::transformed(plane, singular), std::invalid_argument);
  auto singular_future = tf::cpp::async::transformed(plane, singular);
  CHECK_THROWS_AS(singular_future.get(), std::invalid_argument);
}

TEMPLATE_TEST_CASE("transformed rejects invalid and mismatched matrices",
                   "[cpp][spatial][transformed][validation]", float, double) {
  const auto point2 =
      primitive<TestType, 2>(tf::cpp::primitive_kind::point, {1, 0}, {2});
  const auto point3 =
      primitive<TestType, 3>(tf::cpp::primitive_kind::point, {1, 0, 0}, {3});
  const auto matrix2 = identity_matrix<TestType, 2>();
  const auto matrix3 = identity_matrix<TestType, 3>();

  CHECK_THROWS_AS(tf::cpp::transformed(point2, matrix3), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::transformed(point3, matrix2), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::transformed(point2, tf::cpp::nd_array<TestType>{}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::transformed(
          point3, make_array<TestType>({1, 0, 0, 0, 1, 0, 0, 0, 1}, {9})),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE("Python point batches preserve metadata and all values",
                   "[cpp][spatial][transformed][python-parity][batch][point]",
                   float, double) {
  const auto points3 = primitive<TestType, 3>(
      tf::cpp::primitive_kind::point, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3});
  const auto result3 =
      tf::cpp::transformed(points3, rotation_translation_z_3d<TestType>());
  check_batch_metadata(result3, tf::cpp::primitive_kind::point, 3, {3, 3});
  check_values(result3.data(), {0, 1, 0, -1, 0, 0, 0, 0, 1});

  const auto points2 = primitive<TestType, 2>(tf::cpp::primitive_kind::point,
                                              {1, 0, 0, 1}, {2, 2});
  const auto result2 =
      tf::cpp::transformed(points2, rotation_translation_2d<TestType>(2, 3));
  check_batch_metadata(result2, tf::cpp::primitive_kind::point, 2, {2, 2});
  check_values(result2.data(), {2, 4, 1, 3});
}

TEMPLATE_TEST_CASE(
    "Python 3D primitive batches preserve metadata and all values",
    "[cpp][spatial][transformed][python-parity][batch]", float, double) {
  const auto rotation = rotation_translation_z_3d<TestType>();

  const auto segments =
      primitive<TestType, 3>(tf::cpp::primitive_kind::segment,
                             {1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 2, 0}, {2, 2, 3});
  const auto segment_result = tf::cpp::transformed(segments, rotation);
  check_batch_metadata(segment_result, tf::cpp::primitive_kind::segment, 2,
                       {2, 2, 3});
  check_values(segment_result.data(), {0, 1, 0, 0, 2, 0, -1, 0, 0, -2, 0, 0});

  const auto rays =
      primitive<TestType, 3>(tf::cpp::primitive_kind::ray,
                             {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0}, {2, 2, 3});
  const auto ray_result = tf::cpp::transformed(rays, rotation);
  check_batch_metadata(ray_result, tf::cpp::primitive_kind::ray, 2, {2, 2, 3});
  check_values(ray_result.data(), {0, 0, 0, 0, 1, 0, 0, 1, 0, -1, 0, 0});
  for (int index = 0; index < ray_result.count(); ++index) {
    const auto offset = static_cast<std::size_t>(index) * 6 + 3;
    CHECK(std::hypot(ray_result.data()[offset], ray_result.data()[offset + 1],
                     ray_result.data()[offset + 2]) ==
          Catch::Approx(1).margin(tolerance<TestType>()));
  }

  const auto lines =
      primitive<TestType, 3>(tf::cpp::primitive_kind::line,
                             {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1}, {2, 2, 3});
  const auto line_result = tf::cpp::transformed(lines, rotation);
  check_batch_metadata(line_result, tf::cpp::primitive_kind::line, 2,
                       {2, 2, 3});
  check_values(line_result.data(), {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1});

  const auto boxes =
      primitive<TestType, 3>(tf::cpp::primitive_kind::aabb,
                             {0, 0, 0, 1, 1, 1, 2, 0, 0, 3, 1, 1}, {2, 2, 3});
  const auto box_result =
      tf::cpp::transformed(boxes, translation_3d<TestType>(10, 0, 0));
  check_batch_metadata(box_result, tf::cpp::primitive_kind::aabb, 2, {2, 2, 3});
  check_values(box_result.data(), {10, 0, 0, 11, 1, 1, 12, 0, 0, 13, 1, 1});

  const auto polygons = primitive<TestType, 3>(
      tf::cpp::primitive_kind::polygon,
      {1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0}, {2, 3, 3});
  const auto polygon_result = tf::cpp::transformed(polygons, rotation);
  check_batch_metadata(polygon_result, tf::cpp::primitive_kind::polygon, 2,
                       {2, 3, 3});
  CHECK(polygon_result.polygon_vertex_count() == 3);
  check_values(polygon_result.data(),
               {0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0});

  const auto planes = primitive<TestType, 3>(tf::cpp::primitive_kind::plane,
                                             {0, 0, 1, 0, 0, 0, 1, -5}, {2, 4});
  const auto plane_result =
      tf::cpp::transformed(planes, translation_3d<TestType>(0, 0, 10));
  check_batch_metadata(plane_result, tf::cpp::primitive_kind::plane, 2, {2, 4});
  check_values(plane_result.data(), {0, 0, 1, -10, 0, 0, 1, -15});
}

TEMPLATE_TEST_CASE("rigid transformed point batches preserve distances",
                   "[cpp][spatial][transformed][python-parity][distance]",
                   float, double) {
  const auto points2 = primitive<TestType, 2>(tf::cpp::primitive_kind::point,
                                              {0, 0, 3, 0}, {2, 2});
  const auto result2 = tf::cpp::transformed(
      points2, rotation_translation_45_2d<TestType>(10, 20));
  const auto original2 = std::hypot(points2.data()[2] - points2.data()[0],
                                    points2.data()[3] - points2.data()[1]);
  const auto transformed2 = std::hypot(result2.data()[2] - result2.data()[0],
                                       result2.data()[3] - result2.data()[1]);
  CHECK(transformed2 == Catch::Approx(original2).margin(1e-4));

  const auto points3 = primitive<TestType, 3>(tf::cpp::primitive_kind::point,
                                              {0, 0, 0, 3, 0, 0}, {2, 3});
  const auto result3 = tf::cpp::transformed(
      points3, rotation_translation_45_z_3d<TestType>(10, 20, 30));
  const auto original3 = std::hypot(points3.data()[3] - points3.data()[0],
                                    points3.data()[4] - points3.data()[1],
                                    points3.data()[5] - points3.data()[2]);
  const auto transformed3 = std::hypot(result3.data()[3] - result3.data()[0],
                                       result3.data()[4] - result3.data()[1],
                                       result3.data()[5] - result3.data()[2]);
  CHECK(transformed3 == Catch::Approx(original3).margin(1e-4));
}

TEMPLATE_TEST_CASE(
    "transformed preserves every empty batch shape including closed enum kinds",
    "[cpp][spatial][transformed][empty][enum]", float, double) {
  check_empty_batches<TestType, 2>();
  check_empty_batches<TestType, 3>();
}

TEMPLATE_TEST_CASE(
    "synchronous transformed leaves inputs unchanged and detaches output",
    "[cpp][spatial][transformed][ownership][sync]", float, double) {
  check_detached_output<TestType, 2>();
  check_detached_output<TestType, 3>();
}

TEST_CASE("transformed parallel direction failures propagate synchronously and "
          "asynchronously",
          "[cpp][spatial][transformed][parallel][validation]") {
  constexpr int count = 4096;
  std::vector<float> values(static_cast<std::size_t>(count) * 4, 0.0F);
  const auto rays = tf::cpp::primitive<float, 2>(
      tf::cpp::primitive_kind::ray, make_array(values, {count, 2, 2}));
  const auto identity = identity_matrix<float, 2>();
  CHECK_THROWS_AS(tf::cpp::transformed(rays, identity), std::invalid_argument);
  auto future = tf::cpp::async::transformed(rays, identity);
  CHECK_THROWS_AS(future.get(), std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "sync future and custom resolver transformed results own their storage",
    "[cpp][spatial][transformed][ownership][async]", float, double) {
  check_async_ownership<TestType, 2>();
  check_async_ownership<TestType, 3>();
}
