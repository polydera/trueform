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
#include "trueform/cpp/spatial/async/closest_metric_point.hpp"
#include "trueform/cpp/spatial/closest_metric_point.hpp"
#include "trueform/cpp/spatial/async/closest_metric_point_pair.hpp"
#include "trueform/cpp/spatial/closest_metric_point_pair.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <future>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace {

struct counting_resolver {
  int *count;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    ++*count;
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

template <typename Real, std::size_t Dims = 3>
auto point(std::initializer_list<Real> values)
    -> tf::cpp::primitive<Real, Dims> {
  return tf::cpp::primitive<Real, Dims>(
      tf::cpp::primitive_kind::point,
      make_array<Real>(values, {static_cast<int>(Dims)}));
}

template <typename Real, std::size_t Dims = 3>
auto point_batch(std::initializer_list<Real> values, int count)
    -> tf::cpp::primitive<Real, Dims> {
  return tf::cpp::primitive<Real, Dims>(
      tf::cpp::primitive_kind::point,
      make_array<Real>(values, {count, static_cast<int>(Dims)}));
}

template <typename Real, std::size_t Dims = 3>
auto origin_primitive(tf::cpp::primitive_kind kind)
    -> tf::cpp::primitive<Real, Dims> {
  using primitive_type = tf::cpp::primitive<Real, Dims>;
  switch (kind) {
  case tf::cpp::primitive_kind::point:
    if constexpr (Dims == 2)
      return point<Real, Dims>({0, 0});
    else
      return point<Real, Dims>({0, 0, 0});
  case tf::cpp::primitive_kind::segment:
    if constexpr (Dims == 2)
      return primitive_type(kind, make_array<Real>({-1, 0, 1, 0}, {2, 2}));
    else
      return primitive_type(kind,
                            make_array<Real>({-1, 0, 0, 1, 0, 0}, {2, 3}));
  case tf::cpp::primitive_kind::triangle:
    if constexpr (Dims == 2)
      return primitive_type(kind,
                            make_array<Real>({-1, -1, 1, -1, 0, 1}, {3, 2}));
    else
      return primitive_type(
          kind, make_array<Real>({-1, -1, 0, 1, -1, 0, 0, 1, 0}, {3, 3}));
  case tf::cpp::primitive_kind::ray:
  case tf::cpp::primitive_kind::line:
    if constexpr (Dims == 2)
      return primitive_type(kind, make_array<Real>({0, 0, 1, 0}, {2, 2}));
    else
      return primitive_type(kind, make_array<Real>({0, 0, 0, 1, 0, 0}, {2, 3}));
  case tf::cpp::primitive_kind::plane:
    if constexpr (Dims == 2)
      return primitive_type(kind, make_array<Real>({0, 1, 0}, {3}));
    else
      return primitive_type(kind, make_array<Real>({0, 0, 1, 0}, {4}));
  case tf::cpp::primitive_kind::aabb:
    if constexpr (Dims == 2)
      return primitive_type(kind, make_array<Real>({-1, -1, 1, 1}, {2, 2}));
    else
      return primitive_type(kind,
                            make_array<Real>({-1, -1, -1, 1, 1, 1}, {2, 3}));
  case tf::cpp::primitive_kind::polygon:
    if constexpr (Dims == 2)
      return primitive_type(
          kind, make_array<Real>({-1, -1, 1, -1, 1, 1, -1, 1}, {4, 2}));
    else
      return primitive_type(
          kind,
          make_array<Real>({-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0}, {4, 3}));
  case tf::cpp::primitive_kind::vector:
    if constexpr (Dims == 2)
      return primitive_type(kind, make_array<Real>({1, 0}, {2}));
    else
      return primitive_type(kind, make_array<Real>({1, 0, 0}, {3}));
  }
  throw std::logic_error("unknown primitive kind");
}

template <typename Real>
auto single_point_result(const tf::cpp::primitive<Real> &a,
                         const tf::cpp::primitive<Real> &b)
    -> tf::cpp::closest_metric_point_result<Real> {
  return std::get<tf::cpp::closest_metric_point_result<Real>>(
      tf::cpp::closest_metric_point(a, b));
}

template <typename Real, std::size_t Dims = 3>
auto single_pair_result(const tf::cpp::primitive<Real, Dims> &a,
                        const tf::cpp::primitive<Real, Dims> &b)
    -> tf::cpp::closest_metric_point_pair_result<Real> {
  return std::get<tf::cpp::closest_metric_point_pair_result<Real>>(
      tf::cpp::closest_metric_point_pair(a, b));
}

template <typename T>
auto check_shape(const tf::cpp::nd_array<T> &array,
                 std::initializer_list<int> shape) -> void {
  REQUIRE(array.ndim() == static_cast<int>(shape.size()));
  int dimension = 0;
  for (const auto size : shape)
    CHECK(array.shape_at(dimension++) == size);
}

template <typename A, typename B, typename = void>
struct has_closest_metric_point_pair : std::false_type {};

template <typename A, typename B>
struct has_closest_metric_point_pair<
    A, B,
    std::void_t<decltype(tf::cpp::closest_metric_point_pair(
        std::declval<const A &>(), std::declval<const B &>()))>>
    : std::true_type {};

template <typename A, typename B, typename = void>
struct has_async_closest_metric_point_pair : std::false_type {};

template <typename A, typename B>
struct has_async_closest_metric_point_pair<
    A, B,
    std::void_t<decltype(tf::cpp::async::closest_metric_point_pair(
        std::declval<const A &>(), std::declval<const B &>()))>>
    : std::true_type {};

template <typename Real0, typename Real1>
using closest_pair_2d_function = std::variant<
    tf::cpp::closest_metric_point_pair_result<std::common_type_t<Real0, Real1>>,
    tf::cpp::closest_metric_point_pair_batch_result<
        std::common_type_t<Real0, Real1>>> (*)(
    const tf::cpp::primitive<Real0, 2> &, const tf::cpp::primitive<Real1, 2> &);

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

} // namespace

TEMPLATE_TEST_CASE(
    "2D closest metric point pair covers all ordered spatial kinds",
    "[cpp][spatial][closest_pair][2d][dispatch]", float, double) {
  constexpr std::array<std::size_t, 7> strides{2, 4, 6, 4, 4, 4, 8};

  for (std::size_t index_a = 0; index_a < spatial_kinds_2d.size(); ++index_a) {
    const auto kind_a = spatial_kinds_2d[index_a];
    const auto a = origin_primitive<TestType, 2>(kind_a);
    CHECK(a.element_stride() == strides[index_a]);
    for (std::size_t index_b = 0; index_b < spatial_kinds_2d.size();
         ++index_b) {
      const auto kind_b = spatial_kinds_2d[index_b];
      INFO("ordered 2D primitive pair " << static_cast<int>(kind_a) << " x "
                                        << static_cast<int>(kind_b));
      const auto b = origin_primitive<TestType, 2>(kind_b);
      CHECK(b.element_stride() == strides[index_b]);
      const auto pair = single_pair_result(a, b);
      check_shape(pair.point0, {2});
      check_shape(pair.point1, {2});
      CHECK(pair.distance2 == Catch::Approx(0).margin(1e-5));
      CHECK(pair.point0[0] == Catch::Approx(pair.point1[0]).margin(1e-5));
      CHECK(pair.point0[1] == Catch::Approx(pair.point1[1]).margin(1e-5));
    }
  }

  const auto scalar = point<TestType, 2>({0, 0});
  const auto vector =
      origin_primitive<TestType, 2>(tf::cpp::primitive_kind::vector);
  CHECK_THROWS_AS(tf::cpp::closest_metric_point_pair(vector, scalar),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::closest_metric_point_pair(scalar, vector),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      (origin_primitive<TestType, 2>(tf::cpp::primitive_kind::plane)),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE("2D closest metric point pair preserves scalar broadcast "
                   "pairwise and empty modes",
                   "[cpp][spatial][closest_pair][2d][batch]", float, double) {
  using scalar_result = tf::cpp::closest_metric_point_pair_result<TestType>;
  using batch_result =
      tf::cpp::closest_metric_point_pair_batch_result<TestType>;

  const auto scalar = point<TestType, 2>({3, 0});
  const auto a = point_batch<TestType, 2>({0, 0, 10, 0}, 2);
  const auto b = point_batch<TestType, 2>({1, 0, 8, 0}, 2);
  REQUIRE(scalar.element_stride() == 2);
  REQUIRE(a.element_stride() == 2);
  REQUIRE(b.element_stride() == 2);

  const auto scalar_scalar = tf::cpp::closest_metric_point_pair(scalar, scalar);
  REQUIRE(std::holds_alternative<scalar_result>(scalar_scalar));
  const auto &single = std::get<scalar_result>(scalar_scalar);
  check_shape(single.point0, {2});
  check_shape(single.point1, {2});
  CHECK(single.distance2 == TestType{0});

  const auto batch_scalar =
      std::get<batch_result>(tf::cpp::closest_metric_point_pair(a, scalar));
  const auto scalar_batch =
      std::get<batch_result>(tf::cpp::closest_metric_point_pair(scalar, a));
  const auto pairwise =
      std::get<batch_result>(tf::cpp::closest_metric_point_pair(a, b));

  check_shape(batch_scalar.points0, {2, 2});
  check_shape(batch_scalar.points1, {2, 2});
  check_shape(batch_scalar.distances, {2});
  CHECK(batch_scalar.points0[0] == TestType{0});
  CHECK(batch_scalar.points0[2] == TestType{10});
  CHECK(batch_scalar.points1[0] == TestType{3});
  CHECK(batch_scalar.points1[2] == TestType{3});
  CHECK(batch_scalar.distances[0] == TestType{9});
  CHECK(batch_scalar.distances[1] == TestType{49});

  check_shape(scalar_batch.points0, {2, 2});
  check_shape(scalar_batch.points1, {2, 2});
  CHECK(scalar_batch.points0[0] == TestType{3});
  CHECK(scalar_batch.points0[2] == TestType{3});
  CHECK(scalar_batch.points1[0] == TestType{0});
  CHECK(scalar_batch.points1[2] == TestType{10});
  CHECK(scalar_batch.distances[0] == TestType{9});
  CHECK(scalar_batch.distances[1] == TestType{49});

  check_shape(pairwise.points0, {2, 2});
  check_shape(pairwise.points1, {2, 2});
  CHECK(pairwise.points0[0] == TestType{0});
  CHECK(pairwise.points0[2] == TestType{10});
  CHECK(pairwise.points1[0] == TestType{1});
  CHECK(pairwise.points1[2] == TestType{8});
  CHECK(pairwise.distances[0] == TestType{1});
  CHECK(pairwise.distances[1] == TestType{4});

  const auto empty = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::point, make_empty<TestType>({0, 2}));
  REQUIRE(empty.element_stride() == 2);
  for (const auto &result :
       {tf::cpp::closest_metric_point_pair(empty, scalar),
        tf::cpp::closest_metric_point_pair(scalar, empty),
        tf::cpp::closest_metric_point_pair(empty, empty)}) {
    const auto &batch = std::get<batch_result>(result);
    check_shape(batch.points0, {0, 2});
    check_shape(batch.points1, {0, 2});
    check_shape(batch.distances, {0});
    CHECK(batch.points0.empty());
    CHECK(batch.points1.empty());
    CHECK(batch.distances.empty());
  }

  const auto unequal = point_batch<TestType, 2>({0, 0, 1, 0, 2, 0}, 3);
  CHECK_THROWS_AS(tf::cpp::closest_metric_point_pair(a, unequal),
                  std::invalid_argument);
}

TEST_CASE("2D closest metric point pair preserves mixed precision orders",
          "[cpp][spatial][closest_pair][2d][precision]") {
  using exact_result =
      std::variant<tf::cpp::closest_metric_point_pair_result<double>,
                   tf::cpp::closest_metric_point_pair_batch_result<double>>;

  const auto point32 = point<float, 2>({1, 2});
  const auto point64 = point<double, 2>({4, 6});
  const auto float_double =
      tf::cpp::closest_metric_point_pair(point32, point64);
  const auto double_float =
      tf::cpp::closest_metric_point_pair(point64, point32);
  STATIC_REQUIRE(std::is_same_v<decltype(float_double), const exact_result>);
  STATIC_REQUIRE(std::is_same_v<decltype(double_float), const exact_result>);

  const auto &forward =
      std::get<tf::cpp::closest_metric_point_pair_result<double>>(float_double);
  const auto &reverse =
      std::get<tf::cpp::closest_metric_point_pair_result<double>>(double_float);
  check_shape(forward.point0, {2});
  check_shape(forward.point1, {2});
  CHECK(forward.point0[0] == 1.0);
  CHECK(forward.point1[0] == 4.0);
  CHECK(forward.distance2 == 25.0);
  CHECK(reverse.point0[0] == 4.0);
  CHECK(reverse.point1[0] == 1.0);
  CHECK(reverse.distance2 == 25.0);

  const auto batch32 = point_batch<float, 2>({0, 0, 10, 0}, 2);
  const auto batch64 = point_batch<double, 2>({0, 0, 10, 0}, 2);
  const auto float_batch = tf::cpp::closest_metric_point_pair(point64, batch32);
  const auto double_batch =
      tf::cpp::closest_metric_point_pair(point32, batch64);
  STATIC_REQUIRE(std::is_same_v<decltype(float_batch), const exact_result>);
  STATIC_REQUIRE(std::is_same_v<decltype(double_batch), const exact_result>);
  const auto &float_values =
      std::get<tf::cpp::closest_metric_point_pair_batch_result<double>>(
          float_batch);
  const auto &double_values =
      std::get<tf::cpp::closest_metric_point_pair_batch_result<double>>(
          double_batch);
  check_shape(float_values.points0, {2, 2});
  check_shape(double_values.points1, {2, 2});
  CHECK(float_values.distances[0] == 52.0);
  CHECK(float_values.distances[1] == 72.0);
  CHECK(double_values.distances[0] == 5.0);
  CHECK(double_values.distances[1] == 85.0);
}

TEST_CASE("2D closest metric point pair links all additive archive symbols",
          "[cpp][spatial][closest_pair][2d][archive-link]") {
  const closest_pair_2d_function<float, float> float_float =
      &tf::cpp::closest_metric_point_pair<float, float, 2>;
  const closest_pair_2d_function<double, double> double_double =
      &tf::cpp::closest_metric_point_pair<double, double, 2>;
  const closest_pair_2d_function<float, double> float_double =
      &tf::cpp::closest_metric_point_pair<float, double, 2>;
  const closest_pair_2d_function<double, float> double_float =
      &tf::cpp::closest_metric_point_pair<double, float, 2>;

  const auto point32 = point<float, 2>({1, 2});
  const auto point64 = point<double, 2>({4, 6});
  CHECK(std::get<tf::cpp::closest_metric_point_pair_result<float>>(
            float_float(point32, point32))
            .distance2 == 0.0f);
  CHECK(std::get<tf::cpp::closest_metric_point_pair_result<double>>(
            double_double(point64, point64))
            .distance2 == 0.0);
  CHECK(std::get<tf::cpp::closest_metric_point_pair_result<double>>(
            float_double(point32, point64))
            .distance2 == 25.0);
  CHECK(std::get<tf::cpp::closest_metric_point_pair_result<double>>(
            double_float(point64, point32))
            .distance2 == 25.0);
}

TEST_CASE("2D closest metric point pair rejects dimensional mismatches",
          "[cpp][spatial][closest_pair][2d][compile-time]") {
  using point2f = tf::cpp::primitive<float, 2>;
  using point2d = tf::cpp::primitive<double, 2>;
  using point3f = tf::cpp::primitive<float, 3>;
  using point3d = tf::cpp::primitive<double, 3>;

  STATIC_REQUIRE(has_closest_metric_point_pair<point2f, point2d>::value);
  STATIC_REQUIRE(has_async_closest_metric_point_pair<point2f, point2d>::value);
  STATIC_REQUIRE_FALSE(has_closest_metric_point_pair<point2f, point3d>::value);
  STATIC_REQUIRE_FALSE(has_closest_metric_point_pair<point3f, point2d>::value);
  STATIC_REQUIRE_FALSE(
      has_async_closest_metric_point_pair<point2f, point3d>::value);
  STATIC_REQUIRE_FALSE(
      has_async_closest_metric_point_pair<point3f, point2d>::value);

  STATIC_REQUIRE(has_closest_metric_point_pair<point3f, point3d>::value);
  STATIC_REQUIRE(has_async_closest_metric_point_pair<point3f, point3d>::value);
  STATIC_REQUIRE(std::is_same_v<tf::cpp::primitive<float>, point3f>);
}

TEMPLATE_TEST_CASE("async 2D closest metric point pair preserves exact futures "
                   "resolver ownership and errors",
                   "[cpp][spatial][closest_pair][2d][async]", float, double) {
  using exact_result =
      std::variant<tf::cpp::closest_metric_point_pair_result<TestType>,
                   tf::cpp::closest_metric_point_pair_batch_result<TestType>>;

  auto owned = [] {
    const auto lhs = point<TestType, 2>({1, 2});
    const auto rhs = point<TestType, 2>({4, 6});
    return tf::cpp::async::closest_metric_point_pair(lhs, rhs);
  }();
  STATIC_REQUIRE(std::is_same_v<decltype(owned), std::future<exact_result>>);
  const auto owned_value =
      std::get<tf::cpp::closest_metric_point_pair_result<TestType>>(
          owned.get());
  check_shape(owned_value.point0, {2});
  check_shape(owned_value.point1, {2});
  CHECK(owned_value.point0[0] == TestType{1});
  CHECK(owned_value.point1[0] == TestType{4});
  CHECK(owned_value.distance2 == TestType{25});

  int submissions = 0;
  auto custom = tf::cpp::async::closest_metric_point_pair(
      counting_resolver{&submissions}, point<TestType, 2>({1, 2}),
      point<TestType, 2>({4, 6}));
  STATIC_REQUIRE(std::is_same_v<decltype(custom), std::future<exact_result>>);
  CHECK(submissions == 1);
  CHECK(std::get<tf::cpp::closest_metric_point_pair_result<TestType>>(
            custom.get())
            .distance2 == TestType{25});

  const auto vector =
      origin_primitive<TestType, 2>(tf::cpp::primitive_kind::vector);
  auto vector_failure = tf::cpp::async::closest_metric_point_pair(
      point<TestType, 2>({0, 0}), vector);
  CHECK_THROWS_AS(vector_failure.get(), std::invalid_argument);

  auto unequal_failure = tf::cpp::async::closest_metric_point_pair(
      point_batch<TestType, 2>({0, 0, 1, 0}, 2),
      point_batch<TestType, 2>({0, 0, 1, 0, 2, 0}, 3));
  CHECK_THROWS_AS(unequal_failure.get(), std::invalid_argument);
}

TEST_CASE("async 2D closest metric point pair preserves mixed precision",
          "[cpp][spatial][closest_pair][2d][async][precision]") {
  using exact_result =
      std::variant<tf::cpp::closest_metric_point_pair_result<double>,
                   tf::cpp::closest_metric_point_pair_batch_result<double>>;

  auto float_double = tf::cpp::async::closest_metric_point_pair(
      point<float, 2>({1, 2}), point<double, 2>({4, 6}));
  auto double_float = tf::cpp::async::closest_metric_point_pair(
      point<double, 2>({4, 6}), point<float, 2>({1, 2}));
  STATIC_REQUIRE(
      std::is_same_v<decltype(float_double), std::future<exact_result>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(double_float), std::future<exact_result>>);
  CHECK(std::get<tf::cpp::closest_metric_point_pair_result<double>>(
            float_double.get())
            .distance2 == 25.0);
  CHECK(std::get<tf::cpp::closest_metric_point_pair_result<double>>(
            double_float.get())
            .distance2 == 25.0);
}

TEMPLATE_TEST_CASE(
    "2D polygon segment closest pair checks the skipped polygon vertex",
    "[cpp][spatial][closest_pair][2d][regression]", float, double) {
  const auto polygon = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::polygon,
      make_array<TestType>({0, 0, 4, 0, 4, 4, 0, 4}, {4, 2}));
  const auto segment = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::segment,
      make_array<TestType>({-2, 1, 1, -2}, {2, 2}));

  const auto forward = single_pair_result(polygon, segment);
  CHECK(forward.distance2 == Catch::Approx(0.5));
  CHECK(forward.point0[0] == Catch::Approx(0));
  CHECK(forward.point0[1] == Catch::Approx(0));
  CHECK(forward.point1[0] == Catch::Approx(-0.5));
  CHECK(forward.point1[1] == Catch::Approx(-0.5));

  const auto reverse = single_pair_result(segment, polygon);
  CHECK(reverse.distance2 == Catch::Approx(0.5));
  CHECK(reverse.point0[0] == Catch::Approx(-0.5));
  CHECK(reverse.point0[1] == Catch::Approx(-0.5));
  CHECK(reverse.point1[0] == Catch::Approx(0));
  CHECK(reverse.point1[1] == Catch::Approx(0));

  const auto concave = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::polygon,
      make_array<TestType>({0, 0, 4, 0, 4, 1, 1, 1, 1, 4, 0, 4}, {6, 2}));
  const auto outside_segment = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::segment,
      make_array<TestType>({2, 2, 3, 2}, {2, 2}));
  const auto outside = single_pair_result(concave, outside_segment);
  CHECK(outside.distance2 == Catch::Approx(1));
  CHECK(outside.point0[0] == Catch::Approx(3));
  CHECK(outside.point0[1] == Catch::Approx(1));
  CHECK(outside.point1[0] == Catch::Approx(3));
  CHECK(outside.point1[1] == Catch::Approx(2));
}

TEMPLATE_TEST_CASE("2D collinear rays preserve overlap and divergence",
                   "[cpp][spatial][closest_pair][2d][regression]", float,
                   double) {
  const auto ahead = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::ray, make_array<TestType>({1, 0, 1, 0}, {2, 2}));
  const auto behind = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::ray, make_array<TestType>({0, 0, 1, 0}, {2, 2}));

  const auto ahead_behind = single_pair_result(ahead, behind);
  CHECK(ahead_behind.distance2 == Catch::Approx(0));
  CHECK(ahead_behind.point0[0] == Catch::Approx(ahead_behind.point1[0]));
  CHECK(ahead_behind.point0[1] == Catch::Approx(ahead_behind.point1[1]));
  CHECK(ahead_behind.point0[0] >= TestType{1});
  CHECK(ahead_behind.point0[1] == TestType{0});

  const auto behind_ahead = single_pair_result(behind, ahead);
  CHECK(behind_ahead.distance2 == Catch::Approx(0));
  CHECK(behind_ahead.point0[0] == Catch::Approx(behind_ahead.point1[0]));
  CHECK(behind_ahead.point0[1] == Catch::Approx(behind_ahead.point1[1]));
  CHECK(behind_ahead.point0[0] >= TestType{1});
  CHECK(behind_ahead.point0[1] == TestType{0});

  const auto opposite_behind = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::ray,
      make_array<TestType>({0, 0, -1, 0}, {2, 2}));
  const auto diverging = single_pair_result(ahead, opposite_behind);
  CHECK(diverging.distance2 == Catch::Approx(1));
  CHECK(diverging.point0[0] == Catch::Approx(1));
  CHECK(diverging.point1[0] == Catch::Approx(0));

  const auto opposite_ahead = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::ray,
      make_array<TestType>({1, 0, -1, 0}, {2, 2}));
  const auto converging = single_pair_result(behind, opposite_ahead);
  CHECK(converging.distance2 == Catch::Approx(0));
  CHECK(converging.point0[0] == Catch::Approx(converging.point1[0]));
  CHECK(converging.point0[1] == Catch::Approx(converging.point1[1]));
}

TEMPLATE_TEST_CASE("closest pair minimizes parallel ray ray boundaries",
                   "[cpp][spatial][closest_pair][regression][ray]", float,
                   double) {
  SECTION("2D axial offset in both orders") {
    const auto ray0 = tf::cpp::primitive<TestType, 2>(
        tf::cpp::primitive_kind::ray,
        make_array<TestType>({0, 0, 1, 0}, {2, 2}));
    const auto ray1 = tf::cpp::primitive<TestType, 2>(
        tf::cpp::primitive_kind::ray,
        make_array<TestType>({5, 2, 1, 0}, {2, 2}));

    const auto forward = single_pair_result(ray0, ray1);
    CHECK(forward.distance2 == Catch::Approx(4));
    CHECK(forward.point0[0] == Catch::Approx(5));
    CHECK(forward.point0[1] == Catch::Approx(0));
    CHECK(forward.point1[0] == Catch::Approx(5));
    CHECK(forward.point1[1] == Catch::Approx(2));

    const auto reverse = single_pair_result(ray1, ray0);
    CHECK(reverse.distance2 == Catch::Approx(4));
    CHECK(reverse.point0[0] == Catch::Approx(5));
    CHECK(reverse.point0[1] == Catch::Approx(2));
    CHECK(reverse.point1[0] == Catch::Approx(5));
    CHECK(reverse.point1[1] == Catch::Approx(0));
  }

  SECTION("3D axial offset in both orders") {
    const auto ray0 = tf::cpp::primitive<TestType>(
        tf::cpp::primitive_kind::ray,
        make_array<TestType>({0, 0, 0, 1, 0, 0}, {2, 3}));
    const auto ray1 = tf::cpp::primitive<TestType>(
        tf::cpp::primitive_kind::ray,
        make_array<TestType>({5, 3, 4, 1, 0, 0}, {2, 3}));

    const auto forward = single_pair_result(ray0, ray1);
    CHECK(forward.distance2 == Catch::Approx(25));
    CHECK(forward.point0[0] == Catch::Approx(5));
    CHECK(forward.point0[1] == Catch::Approx(0));
    CHECK(forward.point0[2] == Catch::Approx(0));
    CHECK(forward.point1[0] == Catch::Approx(5));
    CHECK(forward.point1[1] == Catch::Approx(3));
    CHECK(forward.point1[2] == Catch::Approx(4));

    const auto reverse = single_pair_result(ray1, ray0);
    CHECK(reverse.distance2 == Catch::Approx(25));
    CHECK(reverse.point0[0] == Catch::Approx(5));
    CHECK(reverse.point0[1] == Catch::Approx(3));
    CHECK(reverse.point0[2] == Catch::Approx(4));
    CHECK(reverse.point1[0] == Catch::Approx(5));
    CHECK(reverse.point1[1] == Catch::Approx(0));
    CHECK(reverse.point1[2] == Catch::Approx(0));
  }
}

TEMPLATE_TEST_CASE("closest pair minimizes parallel ray segment boundaries",
                   "[cpp][spatial][closest_pair][regression][ray]", float,
                   double) {
  SECTION("2D axial offset in both orders") {
    const auto ray = tf::cpp::primitive<TestType, 2>(
        tf::cpp::primitive_kind::ray,
        make_array<TestType>({0, 2, 1, 0}, {2, 2}));
    const auto segment = tf::cpp::primitive<TestType, 2>(
        tf::cpp::primitive_kind::segment,
        make_array<TestType>({5, 0, 8, 0}, {2, 2}));

    const auto forward = single_pair_result(ray, segment);
    CHECK(forward.distance2 == Catch::Approx(4));
    CHECK(forward.point0[0] == Catch::Approx(5));
    CHECK(forward.point0[1] == Catch::Approx(2));
    CHECK(forward.point1[0] == Catch::Approx(5));
    CHECK(forward.point1[1] == Catch::Approx(0));

    const auto reverse = single_pair_result(segment, ray);
    CHECK(reverse.distance2 == Catch::Approx(4));
    CHECK(reverse.point0[0] == Catch::Approx(5));
    CHECK(reverse.point0[1] == Catch::Approx(0));
    CHECK(reverse.point1[0] == Catch::Approx(5));
    CHECK(reverse.point1[1] == Catch::Approx(2));
  }

  SECTION("3D axial offset in both orders") {
    const auto ray = tf::cpp::primitive<TestType>(
        tf::cpp::primitive_kind::ray,
        make_array<TestType>({0, 3, 4, 1, 0, 0}, {2, 3}));
    const auto segment = tf::cpp::primitive<TestType>(
        tf::cpp::primitive_kind::segment,
        make_array<TestType>({5, 0, 0, 8, 0, 0}, {2, 3}));

    const auto forward = single_pair_result(ray, segment);
    CHECK(forward.distance2 == Catch::Approx(25));
    CHECK(forward.point0[0] == Catch::Approx(5));
    CHECK(forward.point0[1] == Catch::Approx(3));
    CHECK(forward.point0[2] == Catch::Approx(4));
    CHECK(forward.point1[0] == Catch::Approx(5));
    CHECK(forward.point1[1] == Catch::Approx(0));
    CHECK(forward.point1[2] == Catch::Approx(0));

    const auto reverse = single_pair_result(segment, ray);
    CHECK(reverse.distance2 == Catch::Approx(25));
    CHECK(reverse.point0[0] == Catch::Approx(5));
    CHECK(reverse.point0[1] == Catch::Approx(0));
    CHECK(reverse.point0[2] == Catch::Approx(0));
    CHECK(reverse.point1[0] == Catch::Approx(5));
    CHECK(reverse.point1[1] == Catch::Approx(3));
    CHECK(reverse.point1[2] == Catch::Approx(4));
  }
}

TEMPLATE_TEST_CASE("closest pair minimizes nonparallel rays with both line "
                   "parameters behind",
                   "[cpp][spatial][closest_pair][regression][ray]", float,
                   double) {
  const auto ray0 = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::ray,
      make_array<TestType>({0, 0, -1, 0}, {2, 2}));
  const auto ray1 = tf::cpp::primitive<TestType, 2>(
      tf::cpp::primitive_kind::ray,
      make_array<TestType>({-1, 2, -1, 1}, {2, 2}));

  const auto forward = single_pair_result(ray0, ray1);
  CHECK(forward.distance2 == Catch::Approx(4));
  CHECK(forward.point0[0] == Catch::Approx(-1));
  CHECK(forward.point0[1] == Catch::Approx(0));
  CHECK(forward.point1[0] == Catch::Approx(-1));
  CHECK(forward.point1[1] == Catch::Approx(2));

  const auto reverse = single_pair_result(ray1, ray0);
  CHECK(reverse.distance2 == Catch::Approx(4));
  CHECK(reverse.point0[0] == Catch::Approx(-1));
  CHECK(reverse.point0[1] == Catch::Approx(2));
  CHECK(reverse.point1[0] == Catch::Approx(-1));
  CHECK(reverse.point1[1] == Catch::Approx(0));
}

TEMPLATE_TEST_CASE(
    "closest metric operations preserve values and operand order",
    "[cpp][spatial][closest]", float, double) {
  auto segment = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::segment,
      make_array<TestType>({0, 0, 0, 1, 0, 0}, {2, 3}));
  auto query = point<TestType>({3, 0, 0});

  const auto closest = single_point_result(segment, query);
  check_shape(closest.point, {3});
  CHECK(closest.point[0] == Catch::Approx(1));
  CHECK(closest.point[1] == Catch::Approx(0));
  CHECK(closest.point[2] == Catch::Approx(0));
  CHECK(closest.distance2 == Catch::Approx(4));

  const auto reverse = single_point_result(query, segment);
  CHECK(reverse.point[0] == Catch::Approx(3));
  CHECK(reverse.distance2 == Catch::Approx(4));

  const auto pair = single_pair_result(segment, query);
  check_shape(pair.point0, {3});
  check_shape(pair.point1, {3});
  CHECK(pair.point0[0] == Catch::Approx(1));
  CHECK(pair.point1[0] == Catch::Approx(3));
  CHECK(pair.distance2 == Catch::Approx(4));

  const auto reverse_pair = single_pair_result(query, segment);
  CHECK(reverse_pair.point0[0] == Catch::Approx(3));
  CHECK(reverse_pair.point1[0] == Catch::Approx(1));
  CHECK(reverse_pair.distance2 == Catch::Approx(4));
}

TEMPLATE_TEST_CASE("closest metric dispatch covers all ordered spatial kinds",
                   "[cpp][spatial][closest][dispatch]", float, double) {
  for (const auto kind_a : spatial_kinds) {
    for (const auto kind_b : spatial_kinds) {
      INFO("ordered primitive pair " << static_cast<int>(kind_a) << " x "
                                     << static_cast<int>(kind_b));
      const auto a = origin_primitive<TestType>(kind_a);
      const auto b = origin_primitive<TestType>(kind_b);
      const auto closest = single_point_result(a, b);
      const auto pair = single_pair_result(a, b);
      check_shape(closest.point, {3});
      check_shape(pair.point0, {3});
      check_shape(pair.point1, {3});
      CHECK(closest.distance2 == Catch::Approx(0).margin(1e-5));
      CHECK(pair.distance2 == Catch::Approx(0).margin(1e-5));
    }
  }
}

TEMPLATE_TEST_CASE("closest metric operations support every batch broadcast",
                   "[cpp][spatial][closest][batch]", float, double) {
  const auto scalar = point<TestType>({3, 0, 0});
  const auto a = point_batch<TestType>({0, 0, 0, 10, 0, 0}, 2);
  const auto b = point_batch<TestType>({1, 0, 0, 8, 0, 0}, 2);

  CHECK(std::holds_alternative<tf::cpp::closest_metric_point_result<TestType>>(
      tf::cpp::closest_metric_point(scalar, scalar)));

  const auto batch_single =
      std::get<tf::cpp::closest_metric_point_batch_result<TestType>>(
          tf::cpp::closest_metric_point(a, scalar));
  const auto single_batch =
      std::get<tf::cpp::closest_metric_point_batch_result<TestType>>(
          tf::cpp::closest_metric_point(scalar, a));
  const auto pairwise =
      std::get<tf::cpp::closest_metric_point_batch_result<TestType>>(
          tf::cpp::closest_metric_point(a, b));

  check_shape(batch_single.points, {2, 3});
  check_shape(batch_single.distances, {2});
  CHECK(batch_single.points[0] == Catch::Approx(0));
  CHECK(batch_single.points[3] == Catch::Approx(10));
  CHECK(batch_single.distances[0] == Catch::Approx(9));
  CHECK(batch_single.distances[1] == Catch::Approx(49));
  CHECK(single_batch.points[0] == Catch::Approx(3));
  CHECK(single_batch.points[3] == Catch::Approx(3));
  CHECK(single_batch.distances[0] == Catch::Approx(9));
  CHECK(single_batch.distances[1] == Catch::Approx(49));
  CHECK(pairwise.points[0] == Catch::Approx(0));
  CHECK(pairwise.points[3] == Catch::Approx(10));
  CHECK(pairwise.distances[0] == Catch::Approx(1));
  CHECK(pairwise.distances[1] == Catch::Approx(4));

  const auto pair_batch_single =
      std::get<tf::cpp::closest_metric_point_pair_batch_result<TestType>>(
          tf::cpp::closest_metric_point_pair(a, scalar));
  const auto pair_single_batch =
      std::get<tf::cpp::closest_metric_point_pair_batch_result<TestType>>(
          tf::cpp::closest_metric_point_pair(scalar, a));
  const auto pair_pairwise =
      std::get<tf::cpp::closest_metric_point_pair_batch_result<TestType>>(
          tf::cpp::closest_metric_point_pair(a, b));

  check_shape(pair_batch_single.points0, {2, 3});
  check_shape(pair_batch_single.points1, {2, 3});
  check_shape(pair_batch_single.distances, {2});
  CHECK(pair_batch_single.points0[0] == Catch::Approx(0));
  CHECK(pair_batch_single.points1[0] == Catch::Approx(3));
  CHECK(pair_single_batch.points0[3] == Catch::Approx(3));
  CHECK(pair_single_batch.points1[3] == Catch::Approx(10));
  CHECK(pair_pairwise.points0[3] == Catch::Approx(10));
  CHECK(pair_pairwise.points1[3] == Catch::Approx(8));
  CHECK(pair_pairwise.distances[1] == Catch::Approx(4));
}

TEST_CASE("mixed closest metric operations widen through archive symbols",
          "[cpp][spatial][closest][precision][archive]") {
  const auto a32 = point<float>({1, 2, 3});
  const auto b32 = point<float>({4, 6, 3});
  const auto a64 = point<double>({1, 2, 3});
  const auto b64 = point<double>({4, 6, 3});

  const auto closest_32_64 = tf::cpp::closest_metric_point(a32, b64);
  const auto closest_64_32 = tf::cpp::closest_metric_point(a64, b32);
  const auto pair_32_64 = tf::cpp::closest_metric_point_pair(a32, b64);
  const auto pair_64_32 = tf::cpp::closest_metric_point_pair(a64, b32);

  static_assert(
      std::is_same<
          decltype(closest_32_64),
          const std::variant<
              tf::cpp::closest_metric_point_result<double>,
              tf::cpp::closest_metric_point_batch_result<double>>>::value,
      "mixed closest result must use double");
  const auto &closest =
      std::get<tf::cpp::closest_metric_point_result<double>>(closest_32_64);
  CHECK(closest.point[0] == Catch::Approx(1));
  CHECK(closest.distance2 == Catch::Approx(25));
  CHECK(std::get<tf::cpp::closest_metric_point_result<double>>(closest_64_32)
            .distance2 == Catch::Approx(25));

  const auto &pair =
      std::get<tf::cpp::closest_metric_point_pair_result<double>>(pair_32_64);
  CHECK(pair.point0[0] == Catch::Approx(1));
  CHECK(pair.point1[0] == Catch::Approx(4));
  CHECK(pair.distance2 == Catch::Approx(25));
  CHECK(std::get<tf::cpp::closest_metric_point_pair_result<double>>(pair_64_32)
            .distance2 == Catch::Approx(25));
}

TEST_CASE("closest metric result arrays own storage with stable shapes",
          "[cpp][spatial][closest][ownership]") {
  auto source = make_array<float>({1, 2, 3}, {3});
  auto a = tf::cpp::primitive<float>(tf::cpp::primitive_kind::point, source);
  const auto b = point<float>({4, 6, 3});
  const auto closest = single_point_result(a, b);
  const auto pair = single_pair_result(a, b);

  CHECK(closest.point.raw_data() != source.raw_data());
  CHECK(pair.point0.raw_data() != source.raw_data());
  CHECK(pair.point1.raw_data() != b.data().raw_data());
  source[0] = 99;
  CHECK(closest.point[0] == Catch::Approx(1));
  CHECK(pair.point0[0] == Catch::Approx(1));

  const auto surviving = [] {
    const auto lhs = point<double>({1, 2, 3});
    const auto rhs = point<double>({4, 6, 3});
    return single_pair_result(lhs, rhs);
  }();
  REQUIRE(surviving.point0.is_valid());
  REQUIRE(surviving.point1.is_valid());
  CHECK(surviving.point0[0] == Catch::Approx(1));
  CHECK(surviving.point1[0] == Catch::Approx(4));
}

TEST_CASE("closest metric operations preserve empty batches",
          "[cpp][spatial][closest][empty]") {
  const auto empty = tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                               make_empty<float>({0, 3}));
  const auto scalar = point<float>({1, 2, 3});

  for (const auto &result : {tf::cpp::closest_metric_point(empty, scalar),
                             tf::cpp::closest_metric_point(scalar, empty),
                             tf::cpp::closest_metric_point(empty, empty)}) {
    const auto &batch =
        std::get<tf::cpp::closest_metric_point_batch_result<float>>(result);
    check_shape(batch.points, {0, 3});
    check_shape(batch.distances, {0});
    CHECK(batch.points.empty());
    CHECK(batch.distances.empty());
  }

  for (const auto &result :
       {tf::cpp::closest_metric_point_pair(empty, scalar),
        tf::cpp::closest_metric_point_pair(scalar, empty),
        tf::cpp::closest_metric_point_pair(empty, empty)}) {
    const auto &batch =
        std::get<tf::cpp::closest_metric_point_pair_batch_result<float>>(
            result);
    check_shape(batch.points0, {0, 3});
    check_shape(batch.points1, {0, 3});
    check_shape(batch.distances, {0});
    CHECK(batch.points0.empty());
    CHECK(batch.points1.empty());
    CHECK(batch.distances.empty());
  }
}

TEST_CASE("closest metric operations reject invalid inputs before indexing",
          "[cpp][spatial][closest][validation]") {
  const auto scalar = point<float>({0, 0, 0});
  const auto vector = origin_primitive<float>(tf::cpp::primitive_kind::vector);
  CHECK_THROWS_AS(tf::cpp::closest_metric_point(vector, scalar),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::closest_metric_point(scalar, vector),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::closest_metric_point_pair(vector, scalar),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::closest_metric_point_pair(scalar, vector),
                  std::invalid_argument);

  const auto two = point_batch<float>({0, 0, 0, 1, 0, 0}, 2);
  const auto three = point_batch<float>({0, 0, 0, 1, 0, 0, 2, 0, 0}, 3);
  CHECK_THROWS_AS(tf::cpp::closest_metric_point(two, three),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::closest_metric_point_pair(two, three),
                  std::invalid_argument);

  CHECK_THROWS_AS((tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                             make_array<float>({0, 0}, {2}))),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::primitive<float>(tf::cpp::primitive_kind::segment,
                                 make_array<float>({0, 0, 0, 1, 1, 1}, {6}))),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::primitive<float>(static_cast<tf::cpp::primitive_kind>(999),
                                 make_array<float>({0, 0, 0}, {3}))),
      std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                             tf::cpp::nd_array<float>{})),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE("async closest metric operations preserve exact results",
                   "[cpp][spatial][closest][async]", float, double) {
  using closest_result =
      std::variant<tf::cpp::closest_metric_point_result<TestType>,
                   tf::cpp::closest_metric_point_batch_result<TestType>>;
  using pair_result =
      std::variant<tf::cpp::closest_metric_point_pair_result<TestType>,
                   tf::cpp::closest_metric_point_pair_batch_result<TestType>>;

  const auto segment = tf::cpp::primitive<TestType>(
      tf::cpp::primitive_kind::segment,
      make_array<TestType>({0, 0, 0, 1, 0, 0}, {2, 3}));
  const auto query = point<TestType>({3, 0, 0});
  const auto expected = tf::cpp::closest_metric_point(segment, query);
  const auto expected_pair = tf::cpp::closest_metric_point_pair(query, segment);

  auto closest = tf::cpp::async::closest_metric_point(segment, query);
  int submissions = 0;
  auto pair = tf::cpp::async::closest_metric_point_pair(
      counting_resolver{&submissions}, query, segment);

  static_assert(
      std::is_same_v<decltype(closest), std::future<closest_result>>,
      "default closest async overload preserves its exact future type");
  static_assert(std::is_same_v<decltype(pair), std::future<pair_result>>,
                "custom closest-pair resolver preserves its exact result");
  CHECK(submissions == 1);

  const auto actual =
      std::get<tf::cpp::closest_metric_point_result<TestType>>(closest.get());
  const auto actual_pair =
      std::get<tf::cpp::closest_metric_point_pair_result<TestType>>(pair.get());
  const auto &sync =
      std::get<tf::cpp::closest_metric_point_result<TestType>>(expected);
  const auto &sync_pair =
      std::get<tf::cpp::closest_metric_point_pair_result<TestType>>(
          expected_pair);
  CHECK(actual.point[0] == Catch::Approx(sync.point[0]));
  CHECK(actual.distance2 == Catch::Approx(sync.distance2));
  CHECK(actual_pair.point0[0] == Catch::Approx(sync_pair.point0[0]));
  CHECK(actual_pair.point1[0] == Catch::Approx(sync_pair.point1[0]));
  CHECK(actual_pair.distance2 == Catch::Approx(sync_pair.distance2));
}

TEST_CASE("async closest metric operations preserve batches and mixed order",
          "[cpp][spatial][closest][async][batch][precision]") {
  using closest_result =
      std::variant<tf::cpp::closest_metric_point_result<double>,
                   tf::cpp::closest_metric_point_batch_result<double>>;
  using pair_result =
      std::variant<tf::cpp::closest_metric_point_pair_result<double>,
                   tf::cpp::closest_metric_point_pair_batch_result<double>>;

  const auto scalar32 = point<float>({3, 0, 0});
  const auto batch64 = point_batch<double>({0, 0, 0, 10, 0, 0}, 2);
  auto closest = tf::cpp::async::closest_metric_point(batch64, scalar32);
  auto pair = tf::cpp::async::closest_metric_point_pair(scalar32, batch64);

  static_assert(std::is_same_v<decltype(closest), std::future<closest_result>>);
  static_assert(std::is_same_v<decltype(pair), std::future<pair_result>>);

  const auto closest_batch =
      std::get<tf::cpp::closest_metric_point_batch_result<double>>(
          closest.get());
  const auto pair_batch =
      std::get<tf::cpp::closest_metric_point_pair_batch_result<double>>(
          pair.get());
  check_shape(closest_batch.points, {2, 3});
  CHECK(closest_batch.points[0] == Catch::Approx(0));
  CHECK(closest_batch.points[3] == Catch::Approx(10));
  CHECK(closest_batch.distances[0] == Catch::Approx(9));
  CHECK(closest_batch.distances[1] == Catch::Approx(49));
  CHECK(pair_batch.points0[0] == Catch::Approx(3));
  CHECK(pair_batch.points0[3] == Catch::Approx(3));
  CHECK(pair_batch.points1[0] == Catch::Approx(0));
  CHECK(pair_batch.points1[3] == Catch::Approx(10));
}

TEST_CASE("async closest metric operations retain operands and report errors",
          "[cpp][spatial][closest][async][ownership][validation]") {
  auto closest = [] {
    const auto lhs = point<float>({1, 2, 3});
    const auto rhs = point<float>({4, 6, 3});
    return tf::cpp::async::closest_metric_point(lhs, rhs);
  }();
  auto pair = [] {
    const auto lhs = point<double>({1, 2, 3});
    const auto rhs = point<double>({4, 6, 3});
    return tf::cpp::async::closest_metric_point_pair(lhs, rhs);
  }();

  const auto closest_value =
      std::get<tf::cpp::closest_metric_point_result<float>>(closest.get());
  const auto pair_value =
      std::get<tf::cpp::closest_metric_point_pair_result<double>>(pair.get());
  CHECK(closest_value.point[0] == Catch::Approx(1));
  CHECK(closest_value.distance2 == Catch::Approx(25));
  CHECK(pair_value.point0[0] == Catch::Approx(1));
  CHECK(pair_value.point1[0] == Catch::Approx(4));

  const auto scalar = point<float>({0, 0, 0});
  const auto vector = origin_primitive<float>(tf::cpp::primitive_kind::vector);
  auto closest_failure = tf::cpp::async::closest_metric_point(vector, scalar);
  auto pair_failure = tf::cpp::async::closest_metric_point_pair(scalar, vector);
  CHECK_THROWS_AS(closest_failure.get(), std::invalid_argument);
  CHECK_THROWS_AS(pair_failure.get(), std::invalid_argument);
}
